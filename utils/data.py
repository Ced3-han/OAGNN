import numpy as np
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.Polypeptide import three_to_one, three_to_index, is_aa


def protein_df_to_structures(df):
    units = [x for _, x in df.groupby(df['subunit'])]
    structures = []
    for df in units:
        current_model_id = None
        current_chain_id = None
        current_segid = None
        current_residue_id = None
        current_resname = None
        structure_builder = StructureBuilder()
        structure_builder.init_structure('structure')

        for i, row in df.iterrows():    
            residue_id = (row.hetero, row.residue, row.insertion_code)
            coord = np.array((row.x, row.y, row.z), "f")
            # get rid of whitespace in atom names
            split_list = row.fullname.split()
            if len(split_list) != 1:
                # atom name has internal spaces, e.g. " N B ", so
                # we do not strip spaces
                name = row.fullname
            else:
                # atom name is like " CA ", so we can strip spaces
                name = split_list[0]

            if row.model != current_model_id:
                current_model_id = row.model
                structure_builder.init_model(current_model_id)
            
            if current_segid != row.segid:
                current_segid = row.segid
                structure_builder.init_seg(current_segid)

            if current_chain_id != row.chain:
                current_chain_id = row.chain
                structure_builder.init_chain(current_chain_id)
                current_residue_id = residue_id
                current_resname = row.resname
                structure_builder.init_residue(
                    row.resname, row.hetero, row.residue, row.insertion_code
                )
            elif current_residue_id != residue_id or current_resname != row.resname:
                current_residue_id = residue_id
                current_resname = row.resname
                structure_builder.init_residue(
                    row.resname, row.hetero, row.residue, row.insertion_code
                )

            structure_builder.init_atom(
                name,
                coord,
                row.bfactor,
                row.occupancy,
                row.altloc,
                row.fullname,
                row.serial_number,
                row.element,
            )

        structures.append(structure_builder.get_structure())
    return structures


def structure_to_seq_coords(structure):
    bb_coords = []
    seq = ''
    seq_nb = []
    for i, res in enumerate(structure.get_residues()):
        if not is_aa(res.get_resname()):
            continue
        if res.has_id('N') and res.has_id('CA') and res.has_id('C') and res.has_id('O'):
            bb_coords.append(np.stack([
                res['N'].get_coord(),
                res['CA'].get_coord(),
                res['C'].get_coord(),
                res['O'].get_coord(),
            ]))
            seq += three_to_one(res.get_resname())
            seq_nb.append(three_to_index(res.get_resname()))
    bb_coords = np.stack(bb_coords)
    return bb_coords, seq, seq_nb
