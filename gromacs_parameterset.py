"""
GromacsParameterSet: translate a GROMACS force field (ITP/ATP/RTP) into an
OpenMM XML force field.

Standalone copy of the custom module that used to live inside the metal-toid
env at parmed/gromacs/parameters.py, so it travels with the repo rather than
with a particular ParmEd install.  Only stock ParmEd is required (it uses
parmed's own CPreProcessor for #ifdef/#include handling).  Driven by
build_mosic_xml.py.
"""
import os
import copy
import io
import warnings
import math
import fnmatch
from collections import OrderedDict
from itertools import permutations

from parmed.parameters import ParameterSet
from parmed.topologyobjects import (
    AtomType, BondType, AngleType, DihedralType, DihedralTypeList,
    ImproperType, RBTorsionType, CmapType, NonbondedExceptionType,
    NoUreyBradley
)
from parmed.structure import Structure
from parmed.modeller import ResidueTemplate
from parmed import unit as u
from parmed.gromacs._cpp import CPreProcessor

# read_itp rewrites "#include" lines to this sentinel so that the include is
# resolved by read_itp itself (shared defines, no double-reading) while the
# preprocessor still decides whether the line is inside a live #ifdef branch.
_INCLUDE_SENTINEL = ';@@GMX_INCLUDE@@'


class GromacsParameterSet(ParameterSet):
    """
    Stores a parameter set defined by GROMACS ITP/ATP/RTP files.
    Analogous to CharmmParameterSet but for GROMACS force field formats.
    
    Parameters
    ----------
    *filenames : str
        ITP, ATP, and RTP files to load. File type is determined by extension.
        ``#include`` directives inside ITP files are followed (each file is
        read once), so ``forcefield.itp`` alone pulls in ffnonbonded/ffbonded/
        gb/cmap/nbfix exactly as grompp would.
    defines : dict, optional
        Preprocessor defines, e.g. ``{'HEAVY_H': 1}`` or ``{'FLEXIBLE': 1}``.
        With none given the default GROMACS branch (normal masses, CHARMM
        TIP3P with LJ on H, rigid water) is taken - previously *both* branches
        of every #ifdef were read and the last definition silently won.
    """

    def __init__(self, *args, defines=None):
        super().__init__()
        self.parametersets = []
        self.combining_rule = 'lorentz'  # GROMACS default
        self.defaults = None
        self.residues = OrderedDict()
        self.defines = OrderedDict((k, str(v)) for k, v in (defines or {}).items())
        self._read_itps = set()

        itps, atps, rtps = [], [], []
        for arg in args:
            ext = os.path.splitext(arg)[1].lower()
            if ext == '.itp':
                itps.append(arg)
            elif ext == '.atp':
                atps.append(arg)
            elif ext == '.rtp':
                rtps.append(arg)
            else:
                raise ValueError(f'Unrecognized file type: {arg}')

        # Order matters: ATP first (atom types), then ITP (parameters), then RTP (residues)
        for atp in atps:
            self.read_atp(atp)
        for itp in itps:
            self.read_itp(itp)
        for rtp in rtps:
            self.read_rtp(rtp)

    # ----------------------------------------------------------------
    # ATP reader — atom type names and masses
    # ----------------------------------------------------------------
    def read_atp(self, fname):
        """
        Read a GROMACS atomtypes.atp file.
        Format: name  mass  ; optional comment
        """
        header_comments = []
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(';'):
                    header_comments.append(line[1:].strip())
                    continue

                # Strip inline comments
                if ';' in line:
                    line = line[:line.index(';')].strip()

                words = line.split()
                if len(words) < 2:
                    continue

                name = words[0]
                mass = float(words[1])
                from parmed.periodic_table import element_by_mass, AtomicNum
                element = AtomicNum[element_by_mass(mass)]
                atype = AtomType(name=name, number=-1, mass=mass, atomic_number=element)
                self.atom_types_str[name] = atype

        self.parametersets.append(
            " | ".join(header_comments[:3]) if header_comments else os.path.basename(fname)
        )

    # ----------------------------------------------------------------
    # ITP reader — force field parameter sections
    # ----------------------------------------------------------------
    def read_itp(self, fname):
        """
        Read a GROMACS ITP file containing force field parameter sections.
        Skips structure-specific sections (moleculetype, atoms, bonds, etc.)
        and only reads parameter-definition sections.
        """
        real = os.path.realpath(fname)
        if real in self._read_itps:
            return
        self._read_itps.add(real)

        header_comments = []
        current_section = None
        continuation_line = ''
        # state for [ moleculetype ] blocks (water models, ions)
        mol = None

        # Run the file through ParmEd's C preprocessor so #ifdef/#else/#endif
        # and #define are honoured.  #include lines are swapped for a sentinel
        # comment: the preprocessor drops it when inside a dead branch, and we
        # recurse into read_itp for live ones.
        with open(fname) as raw:
            src = ''.join(
                f'{_INCLUDE_SENTINEL} {l.strip()[len("#include"):].strip()}\n'
                if l.lstrip().startswith('#include') else l
                for l in raw
            )
        cpp = CPreProcessor(io.StringIO(src), defines=self.defines)
        with cpp:
            for line in cpp:
                line = line.strip()
                if line.startswith(_INCLUDE_SENTINEL):
                    inc = line[len(_INCLUDE_SENTINEL):].strip().strip('"<>')
                    inc_path = os.path.join(os.path.dirname(os.path.abspath(fname)), inc)
                    if not os.path.isfile(inc_path):
                        warnings.warn(f'Could not find included file {inc} (from {fname}); skipping')
                        continue
                    self.defines.update(cpp.defines)
                    self.read_itp(inc_path)
                    cpp.defines.update(self.defines)
                    continue
                if not line:
                    continue

                 # Handle line continuation (backslash at end)
                if line.endswith('\\'):
                    continuation_line += line[:-1] + ' '
                    continue
                elif continuation_line:
                    line = continuation_line + line
                    continuation_line = ''
 
                # Capture header comments before first section
                if line.startswith(';') and current_section is None:
                    header_comments.append(line[1:].strip())
                    continue
                if line.startswith(';'):
                    continue

                # Section headers
                if line.startswith('['):
                    current_section = line.split('[')[1].split(']')[0].strip()
                    continue

                # Strip inline comments
                if ';' in line:
                    line = line[:line.index(';')].strip()
                if not line:
                    continue

                # Molecule definitions (water models, ions) -> residue templates
                if current_section == 'moleculetype':
                    mol = self._start_moleculetype(line)
                    continue
                if current_section in ('atoms', 'bonds', 'settles', 'angles',
                                       'virtual_sites3', 'dummies3', 'virtual_sites2'):
                    if mol is not None:
                        self._parse_moleculetype_line(mol, current_section, line)
                    continue
                if current_section in ('pairs', 'dihedrals', 'cmap', 'exclusions',
                                       'system', 'molecules', 'constrainttypes',
                                       'implicit_genborn_params'):
                    continue

                # Parse parameter sections
                if current_section == 'defaults':
                    self._parse_defaults(line)
                elif current_section == 'atomtypes':
                    self._parse_atomtypes(line)
                elif current_section == 'bondtypes':
                    self._parse_bondtypes(line)
                elif current_section == 'angletypes':
                    self._parse_angletypes(line)
                elif current_section == 'dihedraltypes':
                    self._parse_dihedraltypes(line)
                elif current_section == 'nonbond_params':
                    self._parse_nonbond_params(line)
                elif current_section == 'pairtypes':
                    self._parse_pairtypes(line)
                elif current_section == 'cmaptypes':
                    self._parse_cmaptypes(line)

        self.defines.update(cpp.defines)
        self.parametersets.append(
            " | ".join(header_comments[:3]) if header_comments else os.path.basename(fname)
        )

    # ----------------------------------------------------------------
    # RTP reader — residue templates
    # ----------------------------------------------------------------
    def read_rtp(self, fname):
        """
        Read a GROMACS RTP (residue topology) file.
        Handles [ bondedtypes ] header and per-residue blocks with
        [ atoms ], [ bonds ], [ angles ], [ dihedrals ], [ impropers ].
        Cross-residue references (+/-) in impropers are parsed as
        (offset, atom_name) tuples for use in Script-based assignment.
        """
        header_comments = []
        current_residue = None
        current_subsection = None
        # Known subsections inside a residue block
        _subsections = {'atoms', 'bonds', 'angles', 'dihedrals', 'impropers', 'cmap'}
        # Known top-level non-residue sections
        _top_sections = {'bondedtypes'}

        with open(fname) as f:
            for line in f:
                if line.strip().startswith('#'):
                    continue

                line = line.strip()
                if not line:
                    continue
                if line.startswith(';'):
                    if current_residue is None and current_subsection is None:
                        header_comments.append(line[1:].strip())
                    continue

                # Strip inline comments
                if ';' in line:
                    line = line[:line.index(';')].strip()
                if not line:
                    continue

                # Check for section header
                if line.startswith('['):
                    section_name = line.split('[')[1].split(']')[0].strip()

                    if section_name in _subsections:
                        current_subsection = section_name
                    elif section_name in _top_sections:
                        # Top-level section like [ bondedtypes ]; not a residue
                        current_residue = None
                        current_subsection = section_name
                    else:
                        # New residue
                        current_residue = section_name
                        current_subsection = None
                        self.residues[current_residue] = ResidueTemplate(current_residue)
                    continue

                # Skip bondedtypes data (informational; func types are handled
                # by the ITP parser)
                if current_subsection == 'bondedtypes':
                    continue

                if current_residue is None:
                    continue

                res = self.residues[current_residue]
                words = line.split()

                if current_subsection == 'atoms':
                    # name  type  charge  chargegroup
                    if len(words) >= 3:
                        from parmed.topologyobjects import Atom
                        name = words[0]
                        atype = words[1]
                        charge = float(words[2])
                        atom = Atom(name=name, type=atype, charge=charge)
                        res.add_atom(atom)

                elif current_subsection == 'bonds':
                    if len(words) >= 2:
                        a1, a2 = words[0], words[1]
                        if not hasattr(res, '_external_bonds'):
                            res._external_bonds = set()
                        # Handle inter-residue bonds
                        a1_ext = a1.startswith('-') or a1.startswith('+')
                        a2_ext = a2.startswith('-') or a2.startswith('+')
                        if not hasattr(res, '_raw_ext_bonds'):
                            res._raw_ext_bonds = []
                        if a1_ext or a2_ext:
                            res._raw_ext_bonds.append((a1, a2))
                        if a1_ext and not a2_ext:
                            # a2 is local, connects to adjacent residue
                            res._external_bonds.add(a2)
                            if a1.startswith('-'):
                                res.head = res[a2]
                            else:
                                res.tail = res[a2]
                        elif a2_ext and not a1_ext:
                            # a1 is local, connects to adjacent residue
                            res._external_bonds.add(a1)
                            if a2.startswith('-'):
                                res.head = res[a1]
                            else:
                                res.tail = res[a1]
                        elif not a1_ext and not a2_ext:
                            try:
                                res.add_bond(a1, a2)
                            except KeyError:
                                pass  # cross-residue reference

                elif current_subsection == 'impropers':
                    if len(words) >= 4:
                        if not hasattr(res, '_impr'):
                            res._impr = []
                        # Parse +/- cross-residue offsets into (offset, name) tuples
                        improper = []
                        for w in words[:4]:
                            if w.startswith('+'):
                                improper.append((1, w[1:]))
                            elif w.startswith('-'):
                                improper.append((-1, w[1:]))
                            else:
                                improper.append((0, w))
                        res._impr.append(tuple(improper))

                elif current_subsection == 'cmap':
                    if len(words) >= 5:
                        if not hasattr(res, '_cmap'):
                            res._cmap = []
                        # Parse 5 atom names with +/- offsets
                        cmap_atoms = []
                        for w in words[:5]:
                            if w.startswith('+'):
                                cmap_atoms.append((1, w[1:]))
                            elif w.startswith('-'):
                                cmap_atoms.append((-1, w[1:]))
                            else:
                                cmap_atoms.append((0, w))
                        res._cmap.append(tuple(cmap_atoms))

        # Post-process: infer external bonds from cross-residue references.
        # If any residue references +X in a bond, then atom X should be an
        # external bond on any residue type containing it (since that residue
        # could be the "next" one in a chain). Similarly for -X.
        forward_targets = set()   # atom names referenced via + (e.g. 'N' from 'C +N')
        backward_targets = set()  # atom names referenced via - (e.g. 'C' from 'N -C')
        for resname, res in self.residues.items():
            for a1, a2 in getattr(res, '_raw_ext_bonds', []):
                if a1.startswith('+'):
                    forward_targets.add(a1[1:])
                elif a1.startswith('-'):
                    backward_targets.add(a1[1:])
                if a2.startswith('+'):
                    forward_targets.add(a2[1:])
                elif a2.startswith('-'):
                    backward_targets.add(a2[1:])

        all_ext_targets = forward_targets | backward_targets
        for resname, res in self.residues.items():
            if not hasattr(res, '_external_bonds'):
                res._external_bonds = set()
            atom_names = {a.name for a in res.atoms}
            for target in all_ext_targets:
                if target in atom_names:
                    res._external_bonds.add(target)

        self.parametersets.append(
            " | ".join(header_comments[:3]) if header_comments else os.path.basename(fname)
        )

    # ----------------------------------------------------------------
    # [ moleculetype ] blocks -> residue templates (water models, ions)
    # ----------------------------------------------------------------
    def _start_moleculetype(self, line):
        """Begin a new molecule; the template is registered once its [ atoms ]
        section tells us the residue name (SOL, NA, ...)."""
        return {'molname': line.split()[0], 'resname': None, 'template': None,
                'atoms': []}   # atoms: list of (name, type) in index order

    def _parse_moleculetype_line(self, mol, section, line):
        from parmed.topologyobjects import Atom
        words = line.split()
        if section == 'atoms':
            # nr type resnr resname atomname cgnr charge [mass]
            if len(words) < 7:
                return
            if mol['template'] is None:
                mol['resname'] = words[3]
                if mol['resname'] in self.residues:
                    warnings.warn(f"Residue {mol['resname']} defined twice "
                                  f"(moleculetype {mol['molname']}); keeping the first")
                    mol['template'] = False
                    return
                mol['template'] = ResidueTemplate(mol['resname'])
                self.residues[mol['resname']] = mol['template']
            if mol['template'] is False:
                return
            atom = Atom(name=words[4], type=words[1], charge=float(words[6]))
            mol['template'].add_atom(atom)
            mol['atoms'].append((words[4], words[1]))
            return
        if not mol['template']:
            return
        res = mol['template']
        names = [a[0] for a in mol['atoms']]
        types = [a[1] for a in mol['atoms']]

        def idx(w):
            return int(w) - 1

        if section == 'bonds':
            i, j = idx(words[0]), idx(words[1])
            res.add_bond(names[i], names[j])
            # inline parameters (e.g. FLEXIBLE water): register as a bond type
            if len(words) >= 5:
                bt = BondType(float(words[4]) / 4.184 / 100, float(words[3]) * 10)
                self.bond_types.setdefault((types[i], types[j]), bt)
                self.bond_types.setdefault((types[j], types[i]), bt)
        elif section == 'angles':
            i, j, k = idx(words[0]), idx(words[1]), idx(words[2])
            if len(words) >= 6:
                at = AngleType(float(words[5]) / 4.184, float(words[4]))
                self.angle_types.setdefault((types[i], types[j], types[k]), at)
                self.angle_types.setdefault((types[k], types[j], types[i]), at)
                self.urey_bradley_types.setdefault((types[i], types[j], types[k]), NoUreyBradley)
                self.urey_bradley_types.setdefault((types[k], types[j], types[i]), NoUreyBradley)
        elif section == 'settles':
            # OW funct dOH dHH  -> O-H bonds; OpenMM takes rigid-water
            # constraint lengths from the bond/angle equilibrium values, so
            # register zero-force-constant types if none exist.
            o = idx(words[0])
            d_oh, d_hh = float(words[2]) * 10, float(words[3]) * 10
            hs = [k for k in range(len(names)) if k != o][:2]
            for h in hs:
                res.add_bond(names[o], names[h])
                self.bond_types.setdefault((types[o], types[h]), BondType(0.0, d_oh))
                self.bond_types.setdefault((types[h], types[o]), BondType(0.0, d_oh))
            if len(hs) == 2:
                theta = math.degrees(2 * math.asin(d_hh / (2 * d_oh)))
                key = (types[hs[0]], types[o], types[hs[1]])
                self.angle_types.setdefault(key, AngleType(0.0, theta))
                self.angle_types.setdefault(key[::-1], AngleType(0.0, theta))
                self.urey_bradley_types.setdefault(key, NoUreyBradley)
                self.urey_bradley_types.setdefault(key[::-1], NoUreyBradley)
        elif section in ('virtual_sites3', 'dummies3', 'virtual_sites2'):
            warnings.warn(f"Molecule {mol['molname']}: virtual sites are not "
                          "translated; this template will be incomplete")

    # ----------------------------------------------------------------
    # Section parsers for ITP
    # ----------------------------------------------------------------
    def _parse_defaults(self, line):
        """Parse [ defaults ] section: nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ"""
        words = line.split()
        if len(words) < 2:
            raise ValueError('Too few fields in [ defaults ]')
        if words[1] in ('1', '3'):
            self.combining_rule = 'geometric'
        else:
            self.combining_rule = 'lorentz'
        self.defaults = {
            'nbfunc': int(words[0]),
            'comb_rule': int(words[1]),
            'gen_pairs': words[2] if len(words) > 2 else 'no',
            'fudgeLJ': float(words[3]) if len(words) > 3 else 1.0,
            'fudgeQQ': float(words[4]) if len(words) > 4 else 1.0,
        }

    def _parse_atomtypes(self, line):
        """
        Parse [ atomtypes ] section.
        Multiple formats exist:
          name  atnum  mass  charge  ptype  sigma  epsilon
          name  mass  charge  ptype  sigma  epsilon
          name  bond_type  atnum  mass  charge  ptype  sigma  epsilon
        """
        words = line.split()

        # Detect format by finding 'A' or 'V' (ptype column)
        ptype_idx = None
        for j, w in enumerate(words[1:]):
            if w in ('A', 'V', 'S', 'D'):
                ptype_idx = j + 1
                break

        if ptype_idx is None:
            warnings.warn(f'Could not parse atomtype line: {line}')
            return

        name = words[0]
        sigma = float(words[ptype_idx + 1]) * 10      # nm → Å
        epsilon = float(words[ptype_idx + 2])           # kJ/mol
        epsilon /= 4.184  # kJ/mol → kcal/mol
        mass = float(words[ptype_idx - 2])
        charge = float(words[ptype_idx - 1])

        # Try to get atomic number
        atomic_number = -1
        if ptype_idx >= 3:
            try:
                atomic_number = int(words[ptype_idx - 3])
            except ValueError:
                pass

        rmin = sigma * 2**(1/6) / 2  # sigma → rmin/2

        atype = AtomType(name=name, number=-1, mass=mass, atomic_number=atomic_number)
        atype.set_lj_params(epsilon, rmin)
        atype.charge = charge
        self.atom_types_str[name] = atype

    def _parse_bondtypes(self, line):
        """Parse [ bondtypes ]: a1  a2  func  b0  kb"""
        words = line.split()
        a1, a2 = words[0], words[1]
        func = int(words[2])
        if func != 1:
            warnings.warn(f'Bond function {func} not supported, skipping')
            return
        b0 = float(words[3]) * 10          # nm → Å
        kb = float(words[4]) / 4.184 / 100  # kJ/mol/nm² → kcal/mol/Å²

        bt = BondType(kb, b0)
        self.bond_types[(a1, a2)] = bt
        self.bond_types[(a2, a1)] = bt

    def _parse_angletypes(self, line):
        """Parse [ angletypes ]: a1  a2  a3  func  theta0  ktheta  [ub0  kub]"""
        words = line.split()
        a1, a2, a3 = words[0], words[1], words[2]
        func = int(words[3])
        if func not in (1, 5):
            warnings.warn(f'Angle function {func} not supported, skipping')
            return

        theta0 = float(words[4])                # degrees 
        ktheta = float(words[5]) / 4.184      # kJ/mol/rad² → kcal/mol/rad²

        at = AngleType(ktheta, theta0)
        self.angle_types[(a1, a2, a3)] = at
        self.angle_types[(a3, a2, a1)] = at

        # Urey-Bradley (func 5)
        if func == 5 and len(words) >= 8:
            ub0 = float(words[6]) * 10              # nm → Å
            kub = float(words[7]) / 4.184 / 100  # kJ/mol/nm² → kcal/mol/Å²
            ubt = BondType(kub, ub0)
            self.urey_bradley_types[(a1, a2, a3)] = ubt
            self.urey_bradley_types[(a3, a2, a1)] = ubt
        else:
            self.urey_bradley_types[(a1, a2, a3)] = NoUreyBradley
            self.urey_bradley_types[(a3, a2, a1)] = NoUreyBradley

    def _parse_dihedraltypes(self, line):
        """
        Parse [ dihedraltypes ].
        func 1 or 4 = proper periodic dihedral
        func 2      = improper harmonic
        func 3      = Ryckaert-Bellemans
        func 9      = proper periodic (multiple terms)
        """
        words = line.split()

        # Determine if 2-atom or 4-atom key
        # Try to parse words[2] as int — if it works, it's the func (2-atom wildcard form)
        try:
            func = int(words[2])
            a1, a2, a3, a4 = 'X', words[0], words[1], 'X'
            param_start = 3
        except ValueError:
            a1, a2, a3, a4 = words[0], words[1], words[2], words[3]
            func = int(words[4])
            param_start = 5

        key = (a1, a2, a3, a4)
        key = tuple(["" if k == "X" else k for k in key])
        rkey = (a4, a3, a2, a1)
        rkey = tuple(["" if k=="X" else k for k in rkey])
        if func in (1, 4, 9):
            # Proper periodic: phase  kd  periodicity
            phase = float(words[param_start])           # degrees
            kd = float(words[param_start + 1]) / 4.184  # kJ/mol → kcal/mol
            per = int(words[param_start + 2])

            dt = DihedralType(kd, per, phase)

            if func == 9 and key in self.dihedral_types:
                # Multi-term — append
                self.dihedral_types[key].append(dt, override=True)
            else:
                if key not in self.dihedral_types:
                    dtl = DihedralTypeList([dt])
                    self.dihedral_types[key] = dtl
                    self.dihedral_types[rkey] = dtl
                else:
                    # Replace existing term with same periodicity
                    replaced = False
                    for i, existing in enumerate(self.dihedral_types[key]):
                        if existing.per == dt.per:
                            self.dihedral_types[key][i] = dt
                            replaced = True
                            break
                    if not replaced:
                        self.dihedral_types[key].append(dt)

        elif func == 2:
            # Improper harmonic: psi0  kpsi
            psi0 = float(words[param_start])               # degrees
            kpsi = float(words[param_start + 1]) / 4.184  # kJ/mol/rad² → kcal/mol/rad²
            it = ImproperType(kpsi, psi0)
            self.improper_types[key] = it

        elif func == 3:
            # Ryckaert-Bellemans: C0  C1  C2  C3  C4  C5
            coeffs = [float(words[param_start + j]) / 4.184 for j in range(6)]
            rbt = RBTorsionType(*coeffs)
            self.rb_torsion_types[key] = rbt
            self.rb_torsion_types[rkey] = rbt

        else:
            warnings.warn(f'Dihedral function {func} not supported, skipping')

    def _parse_nonbond_params(self, line):
        """Parse [ nonbond_params ]: a1  a2  func  sigma  epsilon"""
        words = line.split()
        a1, a2 = words[0], words[1]
        # func = int(words[2])  # unused
        sig = float(words[3]) * 10           # nm → Å
        eps = float(words[4]) / 4.184     # kJ/mol → kcal/mol
        rmin = sig * 2**(1/6)

        self.nbfix_types[(a1, a2)] = (eps, rmin)
        self.nbfix_types[(a2, a1)] = (eps, rmin)

        if a1 in self.atom_types_str:
            self.atom_types_str[a1].add_nbfix(a2, rmin, eps)
        if a2 in self.atom_types_str:
            self.atom_types_str[a2].add_nbfix(a1, rmin, eps)

    def _parse_pairtypes(self, line):
        """Parse [ pairtypes ]: a1  a2  func  sigma  epsilon"""
        words = line.split()
        a1, a2 = words[0], words[1]
        # func = int(words[2])
        sig = float(words[3]) * 10
        eps = float(words[4]) / 4.184
        rmin = sig * 2**(1/6)
        if a1 == a2:
            # Self-pair — store as nonbonded parameters on the atom type
            if a1 in self.atom_types_str:
                self.atom_types_str[a1].set_lj_params(self.atom_types_str[a1].epsilon, self.atom_types_str[a1].rmin, eps14=eps, rmin14=rmin)
        pt = NonbondedExceptionType(rmin, eps, self.defaults['fudgeQQ'] if self.defaults else 1.0)
        self.pair_types[(a1, a2)] = pt
        self.pair_types[(a2, a1)] = pt

    def _parse_cmaptypes(self, line):
            """
            Parse [ cmaptypes ] entry (after line continuation joining).
            Format: a1 a2 a3 a4 a5 func nx ny val1 val2 ... val(nx*ny)
            Values are in kJ/mol (same as OpenMM, no conversion needed).
            """
            words = line.split()
            if len(words) < 9:
                return
    
            a1, a2, a3, a4, a5 = words[0:5]
            func = int(words[5])
            if func != 1:
                warnings.warn(f'CMAP function {func} not supported, skipping')
                return
    
            nx = int(words[6])
            ny = int(words[7])
            expected = nx * ny
    
            values = []
            for w in words[8:]:
                try:
                    values.append(float(w))
                except ValueError:
                    break
    
            if len(values) != expected:
                warnings.warn(f'CMAP grid incomplete for {a1}-{a2}-{a3}-{a4}-{a5}: '
                            f'expected {expected} values, got {len(values)}')
                return
    
            key = (a1, a2, a3, a4, a5)
            self.cmap_types[key] = {
                'nx': nx, 'ny': ny,
                'values': values
            }
    # ----------------------------------------------------------------
    # Write to OpenMM XML
    # ----------------------------------------------------------------
    # Residues in the CHARMM GROMACS port that are really water models or
    # monatomic ions (from merged.rtp) rather than force-field content.
    WATER_RESIDUES = ('TIP3', 'TP3M', 'HOH', 'SOL', 'SPC', 'SPCE', 'TIP4', 'TIP5')
    ION_RESIDUES = ('SOD', 'POT', 'CLA', 'CAL', 'MG', 'ZN2', 'LIT', 'RUB', 'CES',
                    'BAR', 'CD2', 'OH')

    def write_xml(self, fname, exclude_residues=(), include_residues=None,
                  drop_water_hh_bond=True, classes='all'):
        """
        Write the parameter set to an OpenMM XML force field file.
        Harmonic impropers are handled via <Script>/<InitializationScript>
        blocks that assign impropers at createSystem time, matching the
        approach used by OpenMM's built-in CHARMM force field.

        Parameters
        ----------
        fname : str
            Output XML filename
        exclude_residues : iterable of str
            Residue names (fnmatch patterns allowed) to leave out, e.g.
            ``GromacsParameterSet.WATER_RESIDUES + GromacsParameterSet.ION_RESIDUES``
            to keep the water/ion definitions out of the peptoid XML so they
            can live in their own file.
        include_residues : iterable of str, optional
            If given, only these residues (patterns allowed) are written.
        drop_water_hh_bond : bool
            CHARMM water residues carry an H-H bond (for SHAKE); OpenMM
            templates must not, or PDB/Modeller waters never match. Removes it
            from any 3-atom O/H/H residue.
        classes : 'all' or 'used'
            Which atom classes get <NonbondedForce>/<LennardJonesForce>
            entries. 'used' writes only the classes referenced by the written
            residues, only bonded terms between those classes, and no
            <NBFixPair> entries - use it for a companion water/ion XML that is
            loaded together with the full XML (OpenMM raises on duplicate
            NBFixPair definitions across files, and duplicated torsions would
            be double counted).
        """
        if classes not in ('all', 'used'):
            raise ValueError("classes must be 'all' or 'used'")
        def _selected(resname):
            if include_residues is not None and not any(
                    fnmatch.fnmatchcase(resname, p) for p in include_residues):
                return False
            return not any(fnmatch.fnmatchcase(resname, p) for p in exclude_residues)

        residues = OrderedDict((n, r) for n, r in self.residues.items() if _selected(n))
        skipped = [n for n in self.residues if n not in residues]
        if skipped:
            print(f'Skipping {len(skipped)} residues: {" ".join(skipped)}')

        def _is_water(res):
            els = sorted(self._element_of(a) for a in res.atoms)
            return len(res.atoms) == 3 and els == ['H', 'H', 'O']

        if classes == 'used':
            used = {a.type for r in residues.values() for a in r.atoms}
            nb_types = OrderedDict((n, t) for n, t in self.atom_types_str.items() if n in used)

            def _keep(key):      # bonded term only between written classes
                return all(k in used or k in ('', 'X', None) for k in key)
        else:
            nb_types = self.atom_types_str

            def _keep(key):
                return True

        bond_types = {k: v for k, v in self.bond_types.items() if _keep(k)}
        angle_types = {k: v for k, v in self.angle_types.items() if _keep(k)}
        urey_bradley_types = {k: v for k, v in self.urey_bradley_types.items() if _keep(k)}
        dihedral_types = {k: v for k, v in self.dihedral_types.items() if _keep(k)}
        rb_torsion_types = {k: v for k, v in self.rb_torsion_types.items() if _keep(k)}
        cmap_types = {k: v for k, v in self.cmap_types.items() if _keep(k)}
        improper_types = {k: v for k, v in self.improper_types.items() if _keep(k)}

        with open(fname, 'w') as f:
            f.write('<ForceField>\n')

            # AtomTypes — one per residue-atom, name="{RES}-{ATOM}", class="{gmx_type}"
            from parmed.periodic_table import Element
            f.write(' <AtomTypes>\n')
            written_types = set()
            for resname, res in residues.items():
                for atom in res.atoms:
                    type_name = f'{resname}-{atom.name}'
                    if type_name in written_types:
                        continue
                    written_types.add(type_name)
                    atom_class = atom.type  # GROMACS atom type becomes the class
                    # Look up mass and element from atomtypes
                    atype_data = self.atom_types_str.get(atom_class)
                    if atype_data is not None:
                        mass = atype_data.mass
                        element = ""
                        if hasattr(atype_data, 'atomic_number') and atype_data.atomic_number > 0:
                            try:
                                element = f' element="{Element[atype_data.atomic_number]}"'
                            except (KeyError, IndexError):
                                pass
                    else:
                        mass = 0.0
                        element = ""
                        warnings.warn(f'No atomtype data for class {atom_class} '
                                      f'(used by {type_name})')
                    f.write(f'  <Type name="{type_name}" class="{atom_class}"'
                            f' mass="{mass}"{element}/>\n')
            f.write(' </AtomTypes>\n')

            # Residues from RTP / moleculetypes
            if residues:
                f.write(' <Residues>\n')
                for resname, res in residues.items():
                    water = drop_water_hh_bond and _is_water(res)
                    f.write(f'  <Residue name="{resname}">\n')
                    for atom in res.atoms:
                        type_name = f'{resname}-{atom.name}'
                        f.write(f'   <Atom name="{atom.name}" type="{type_name}"'
                                f' charge="{atom.charge}"/>\n')
                    for bond in res.bonds:
                        if water and self._element_of(bond.atom1) == 'H' \
                                and self._element_of(bond.atom2) == 'H':
                            continue
                        f.write(f'   <Bond atomName1="{bond.atom1.name}"'
                                f' atomName2="{bond.atom2.name}"/>\n')
                    # External bonds from all +/- references
                    ext_bonds = getattr(res, '_external_bonds', set())
                    # Also include head/tail as fallback
                    if res.head is not None:
                        ext_bonds.add(res.head.name)
                    if res.tail is not None:
                        ext_bonds.add(res.tail.name)
                    for ext_atom_name in sorted(ext_bonds):
                        f.write(f'   <ExternalBond atomName="{ext_atom_name}"/>\n')
                    f.write(f'  </Residue>\n')
                f.write(' </Residues>\n')

            # HarmonicBondForce
            if bond_types:
                f.write(' <HarmonicBondForce>\n')
                written = set()
                for key, bt in bond_types.items():
                    if key in written:
                        continue
                    written.add(key)
                    written.add((key[1], key[0]))
                    # Convert back: Å → nm, kcal/mol/Å² → kJ/mol/nm²
                    length_nm = bt.req / 10
                    k_kjnm = bt.k * 4.184 * 100
                    f.write(f'  <Bond class1="{key[0]}" class2="{key[1]}"'
                            f' length="{length_nm:.6f}" k="{k_kjnm:.4f}"/>\n')
                f.write(' </HarmonicBondForce>\n')

            # HarmonicAngleForce
            if angle_types:
                f.write(' <HarmonicAngleForce>\n')
                written = set()
                for key, at in angle_types.items():
                    if key in written:
                        continue
                    written.add(key)
                    written.add((key[2], key[1], key[0]))
                    angle_rad = math.radians(at.theteq)
                    k_kj = at.k * 4.184
                    f.write(f'  <Angle class1="{key[0]}" class2="{key[1]}"'
                            f' class3="{key[2]}" angle="{angle_rad:.10f}"'
                            f' k="{k_kj:.4f}"/>\n')
                f.write(' </HarmonicAngleForce>\n')

            # Urey-Bradley as separate HarmonicBondForce
            ub_entries = []
            written_ub = set()
            for key, ubt in urey_bradley_types.items():
                if key in written_ub:
                    continue
                written_ub.add(key)
                written_ub.add((key[2], key[1], key[0]))
                if ubt is NoUreyBradley:
                    continue
                length_nm = ubt.req / 10
                k_kjnm = ubt.k * 4.184 * 100
                ub_entries.append((key, length_nm, k_kjnm))
            if ub_entries:
                f.write(' <AmoebaUreyBradleyForce>\n')
                for key, length_nm, k_kjnm in ub_entries:
                    k_ub = k_kjnm / 2.0  # GROMACS (1/2)k convention -> OpenMM K convention
                    f.write(f'  <UreyBradley class1="{key[0]}" class2="{key[1]}"'
                            f' class3="{key[2]}" d="{length_nm:.6f}"'
                            f' k="{k_ub:.4f}"/>\n')
                f.write(' </AmoebaUreyBradleyForce>\n')

            # PeriodicTorsionForce (proper dihedrals only)
            if dihedral_types:
                f.write(' <PeriodicTorsionForce>\n')
                written = set()
                for key, dtl in dihedral_types.items():
                    if key in written:
                        continue
                    written.add(key)
                    written.add((key[3], key[2], key[1], key[0]))
                    # Build multi-term proper dihedral
                    attrs = []
                    for j, dt in enumerate(dtl):
                        phase_rad = math.radians(dt.phase)
                        k_kj = dt.phi_k * 4.184
                        attrs.append(
                            f' periodicity{j+1}="{int(dt.per)}"'
                            f' phase{j+1}="{phase_rad:.6f}"'
                            f' k{j+1}="{k_kj:.6f}"'
                        )
                    f.write(f'  <Proper class1="{key[0]}" class2="{key[1]}"'
                            f' class3="{key[2]}" class4="{key[3]}"'
                            + ''.join(attrs) + '/>\n')
                f.write(' </PeriodicTorsionForce>\n')

            # CmapTorsionForce
            if cmap_types:
                f.write(' <CMAPTorsionForce>\n')
                # Write each unique grid as a <Map>, track index
                map_indices = {}
                map_idx = 0
                for key, cmap_data in cmap_types.items():
                    nx, ny = cmap_data['nx'], cmap_data['ny']
                    values = cmap_data['values']
                    map_indices[key] = map_idx

                    # GROMACS → OpenMM convention: transpose + 180 deg phase shift
                    # GROMACS stores psi(rows)×phi(cols) starting at 0 deg
                    # OpenMM expects phi(rows)×psi(cols) starting at -180 deg
                    grid = [values[i*ny:(i+1)*ny] for i in range(nx)]
                    grid = [[grid[j][i] for j in range(nx)] for i in range(ny)]
                    half_x, half_y = nx // 2, ny // 2
                    grid = grid[half_x:] + grid[:half_x]
                    grid = [row[half_y:] + row[:half_y] for row in grid]
                    # Values are in kJ/mol — same units as OpenMM, no conversion
                    flat = [v for row in grid for v in row]
                    val_str = ' '.join(f'{v:.6f}' for v in flat)
                    f.write(f'  <Map>\n{val_str}\n  </Map>\n')
                    map_idx += 1
                # Write Torsion entries referencing the maps by atom class
                for key, cmap_data in cmap_types.items():
                    idx = map_indices[key]
                    f.write(f'  <Torsion map="{idx}"'
                            f' class1="{key[0]}" class2="{key[1]}"'
                            f' class3="{key[2]}" class4="{key[3]}"'
                            f' class5="{key[4]}"/>\n')
                f.write(' </CMAPTorsionForce>\n')

            # RB torsions
            if rb_torsion_types:
                f.write(' <RBTorsionForce>\n')
                written = set()
                for key, rbt in rb_torsion_types.items():
                    if key in written:
                        continue
                    written.add(key)
                    written.add((key[3], key[2], key[1], key[0]))
                    coeffs = [rbt.c0, rbt.c1, rbt.c2, rbt.c3, rbt.c4, rbt.c5]
                    coeffs_kj = [c * 4.184 for c in coeffs]
                    f.write(f'  <Proper class1="{key[0]}" class2="{key[1]}"'
                            f' class3="{key[2]}" class4="{key[3]}"'
                            + ''.join(f' c{i}="{c:.6f}"' for i, c in enumerate(coeffs_kj))
                            + '/>\n')
                f.write(' </RBTorsionForce>\n')

            # NonbondedForce
            fudgeLJ = self.defaults['fudgeLJ'] if self.defaults else 1.0
            fudgeQQ = self.defaults['fudgeQQ'] if self.defaults else 1.0
            f.write(f' <NonbondedForce coulomb14scale="{fudgeQQ}"'
                    f' lj14scale="{fudgeLJ}">\n')
            f.write('  <UseAttributeFromResidue name="charge"/>\n')
            for name, atype in nb_types.items():
                f.write(f'  <Atom class="{name}"'
                        f' sigma="1.0" epsilon="0.0"/>\n')
            f.write(' </NonbondedForce>\n')

            # LennardJonesForce — actual LJ parameters
            f.write(f' <LennardJonesForce lj14scale="{fudgeLJ}">\n')
            for name, atype in nb_types.items():
                if hasattr(atype, 'rmin') and atype.rmin is not None:
                    sigma_nm = (atype.rmin * 2 / 2**(1/6)) / 10
                    eps_kj = abs(atype.epsilon) * 4.184
                else:
                    sigma_nm = 0.0
                    eps_kj = 0.0
                if atype.rmin_14 != atype.rmin or atype.epsilon_14 != atype.epsilon:
                    sigma14_nm = (atype.rmin_14 / 2**(1/6)) / 10
                    eps14_kj = abs(atype.epsilon_14) * 4.184
                    correction = f' sigma14="{sigma14_nm:.8f}" epsilon14="{eps14_kj:.8f}"'
                else:
                    correction = ""
                f.write(f'  <Atom class="{name}"'
                        f' sigma="{sigma_nm:.8f}" epsilon="{eps_kj:.8f}"{correction}/>\n')

            # NBFix pairs (full XML only; duplicates across files are an error)
            if self.nbfix_types and classes == 'all':
                written = set()
                for (a1, a2), (eps, rmin) in self.nbfix_types.items():
                    key = (min(a1, a2), max(a1, a2))
                    if key in written:
                        continue
                    written.add(key)
                    sigma_nm = (rmin / 2**(1/6)) / 10
                    eps_kj = abs(eps) * 4.184
                    f.write(f'  <NBFixPair class1="{a1}" class2="{a2}"'
                            f' sigma="{sigma_nm:.8f}" epsilon="{eps_kj:.8f}"/>\n')

            f.write(' </LennardJonesForce>\n')

            # ----------------------------------------------------------
            # Harmonic impropers via Script (CHARMM-style assignment)
            # ----------------------------------------------------------
            self._write_improper_scripts(f, residues, improper_types)

            f.write('</ForceField>\n')

        print(f'Wrote OpenMM XML force field to {fname}')

    # ----------------------------------------------------------------
    # Script-based harmonic improper generation
    # ----------------------------------------------------------------
    def _element_of(self, atom):
        """Element symbol of a template atom via its GROMACS atom type."""
        from parmed.periodic_table import Element
        atype = self.atom_types_str.get(atom.type)
        if atype is None or atype.atomic_number <= 0:
            return ''
        return Element[atype.atomic_number]

    def _write_improper_scripts(self, f, residues=None, improper_types=None):
        """
        Write a single <Script> block that adds CHARMM-style harmonic
        impropers at createSystem time, matching the approach used by
        OpenMM's built-in CHARMM force field XML.

        The Script defines data structures, a parameter lookup function,
        then iterates over topology residues to resolve atom indices,
        look up parameters, and add torsions to a CustomTorsionForce.
        """
        # Collect residue improper definitions from RTP data
        residue_impropers = {}
        if residues is None:
            residues = self.residues
        if improper_types is None:
            improper_types = self.improper_types
        for resname, res in residues.items():
            if hasattr(res, '_impr') and res._impr:
                residue_impropers[resname] = res._impr

        if not residue_impropers and not improper_types:
            return  # Nothing to write

        # Collect improper parameters, converting to kJ/mol/rad² and radians
        improper_params = {}
        for key, it in improper_types.items():
            k_kj = it.psi_k * 4.184      # kcal/mol/rad² → kJ/mol/rad²
            theta0_rad = math.radians(it.psi_eq)
            improper_params[key] = (k_kj/2, theta0_rad) # Divide by 2 to get proper K from GROMACS (1/2)k convention

        f.write(' <Script>\n')
        f.write('''\
import math
import openmm
import itertools

# --- Embedded data from GROMACS force field ---
# Per-residue improper definitions: {resname: [((offset, name), ...), ...]}
''')

        # Embed residue_impropers dict
        f.write('residue_impropers = {\n')
        for resname, imprs in residue_impropers.items():
            f.write(f'    {resname!r}: (')
            for impr in imprs:
                f.write(f'{impr!r},')
            f.write('),\n')
        f.write('}\n\n')

        # Embed harmonic_improper_types dict
        f.write('# Improper parameters: {(class1,class2,class3,class4): (k_kJ/mol/rad2, theta0_rad)}\n')
        f.write('harmonic_improper_types = {\n')
        for key, (k, t0) in improper_params.items():
            key = tuple([None if k == "" else k for k in key])
            rkey = tuple(reversed(key))
            f.write(f'    {key!r}: (({k:.6f}, {t0:.6f}),),\n')
            f.write(f'    {rkey!r}: (({k:.6f}, {t0:.6f}),),\n')

        f.write('}\n\n')

        # Embed find_improper and main loop using XML-escaped < and >
        f.write('''\
def extract_atom_name(raw_atom_name, templates):
    # Process Drude particles.
    atom_name_prefix = ""
    if raw_atom_name.startswith("Drude-"):
        raw_atom_name = raw_atom_name[len("Drude-"):]
        atom_name_prefix = "D"

    # Return None to indicate an atom without template name information.
    atom_name_parts = raw_atom_name.rsplit("-", maxsplit=1)
    if len(atom_name_parts) &lt; 2:
        return None
    templates.add(atom_name_parts[0])
    return f"{atom_name_prefix}{atom_name_parts[1]}"

def is_valid_improper(atom_indices):
    index_1, index_2, index_3, index_4 = atom_indices
    bonded_to_1 = data.bondedToAtom[index_1]
    bonded_to_4 = data.bondedToAtom[index_4]
    is_improper_1 = index_2 in bonded_to_1 and index_3 in bonded_to_1 and index_4 in bonded_to_1
    is_improper_4 = index_1 in bonded_to_4 and index_2 in bonded_to_4 and index_3 in bonded_to_4
    #return is_improper_1 or is_improper_4
    return True

def find_improper(improper_types, atom_classes):
    # Try all combinations of wildcards.
    for wildcard_count in range(len(atom_classes) + 1):
        for wildcard_indices in itertools.combinations(range(len(atom_classes)), wildcard_count):
            lookup_key = list(atom_classes)
            for wildcard_index in wildcard_indices:
                lookup_key[wildcard_index] = None
            lookup_key = tuple(lookup_key)
            if lookup_key in improper_types:
                return improper_types[lookup_key]
    return None

residue_data = []

for residue_index, residue in enumerate(topology.residues()):
    # Extract the residue or patch names and atom names.
    template_data = templateForResidue[residue_index]
    templates = set()
    if template_data is None:
        atom_names = []
        skip_residue = False
        for atom in residue.atoms():
            atom_name = extract_atom_name(data.atomType[atom], templates)
            if atom_name is None:
                # Skip residues containing atoms without template name data
                # (they might be from, e.g., another solvent force field file).
                skip_residue = True
                break
            atom_names.append(atom_name)
        if skip_residue:
            residue_data.append(([], [], []))
            continue
    else:
        # Extract names from template name.
        for template_index, template_name in enumerate(template_data.name.split("-")):
            if template_index:
                # Patch name.
                templates.add(template_name.rsplit("_", maxsplit=1)[0])
            else:
                # Residue name.
                templates.add(template_name)
        atom_names = [template_data.atoms[data.atomTemplateIndexes[atom]].name for atom in residue.atoms()]

    # Get all unique residue or patch names that have impropers.
    residue_templates = sorted(
        (template for template in templates if template in residue_impropers)
    )

    # Get atom indices and classes for lookup, and ensure atom name uniqueness.
    atom_data = {
        atom_name: (atom.index, self._atomTypes[data.atomType[atom]].atomClass)
        for atom, atom_name in zip(residue.atoms(), atom_names)
    }
    if len(atom_data) != len(atom_names):
        raise ValueError(f"CHARMM: atom name collision in residue with index {residue.index}")

    residue_data.append((atom_data, residue_templates))

def residues_bonded(i, j):
    atoms_j = {index for index, _ in residue_data[j][0].values()}
    return any(neighbor in atoms_j
               for index, _ in residue_data[i][0].values()
               for neighbor in data.bondedToAtom[index])

def find_bonded_residue(i, atom_name):
    # Residue bonded to residue i that contains atom_name and whose atom_name
    # atom is itself bonded into residue i (e.g. CLP of the preceding residue).
    atoms_i = {index for index, _ in residue_data[i][0].values()}
    for j, (atom_data_j, _) in enumerate(residue_data):
        if j == i or atom_name not in atom_data_j:
            continue
        if any(neighbor in atoms_i for neighbor in data.bondedToAtom[atom_data_j[atom_name][0]]):
            return j
    return None

harmonic_impropers = []
for residue_index, (atom_data, residue_templates) in enumerate(residue_data):
    # Determine all impropers to try to add to this residue.  Use a list to
    # try to maintain a consistent order.  There should not be many impropers
    # per residue so this should not be a significant performance issue.
    impropers = []
    for residue in residue_templates:
        for improper in residue_impropers.get(residue, []):
            if improper not in impropers:
                impropers.append(improper)
    for improper in impropers:
        # Find the atoms in the improper.  Skip if any cannot be found.
        atom_indices = []
        atom_classes = []
        skip = False
        log=""
        for offset, atom_name in improper:
            offset_index = residue_index + offset
            # Residues are not always neighbours by index (cyclic molecules,
            # chain breaks), so require an actual bond between the two residues
            # and otherwise look for the bonded residue that carries the atom.
            if offset != 0 and not (0 &lt;= offset_index &lt; len(residue_data)
                                    and residues_bonded(residue_index, offset_index)):
                offset_index = find_bonded_residue(residue_index, atom_name)
            if offset_index is None or not 0 &lt;= offset_index &lt; len(residue_data):
                skip = True
                break
            offset_atom_data = residue_data[offset_index][0]
            if atom_name not in offset_atom_data:
                skip = True
                break
            atom_index, atom_class = offset_atom_data[atom_name]
            atom_indices.append(atom_index)
            atom_classes.append(atom_class)
        if skip:
            continue

        # Skip if this is not a legitimate improper (this could happen if it
        # refers to an adjacent residue in, e.g., an entirely different chain).
        if not is_valid_improper(atom_indices):
            continue
        
        harmonic_improper = find_improper(harmonic_improper_types, atom_classes)
        if harmonic_improper is None:
            raise ValueError(f"No harmonic improper parameters for classes {atom_classes} in residue with index {residue_index} ({residue_templates})")
        else:
            harmonic_impropers.append((*atom_indices,*harmonic_improper))

if harmonic_impropers:
    harmonic_force = openmm.CustomTorsionForce(
        f"k*min(dtheta, 2*pi - dtheta)^2; dtheta = abs(theta - theta0); pi = {math.pi}"
    )
    harmonic_force.addPerTorsionParameter("k")
    harmonic_force.addPerTorsionParameter("theta0")

    for improper in harmonic_impropers:
        harmonic_force.addTorsion(*improper)

    sys.addForce(harmonic_force)
''')
        f.write(' </Script>\n')