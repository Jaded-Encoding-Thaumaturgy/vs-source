from dataclasses import dataclass
from .sector import SectorReadHelper


@dataclass
class CellAdr:
    vob_id: int
    cell_id: int
    start_sector: int
    last_sector: int


@dataclass
class CADT:
    nr_of_vobs: int
    cell_adr_table: list[CellAdr]

    def __init__(self, reader: SectorReadHelper):
        reader._goto_sector_ptr(0x00E0)
        vobcnt, res, end = reader._unpack_byte(2, 2, 4)

        vts_c_adt = []
        cnt = (end + 1 - 6) // 12

        for _ in range(cnt):
            vob_id, cell_id, _res, start_sector, last_sector = reader._unpack_byte(2, 1, 1, 4, 4)
            vts_c_adt += [CellAdr(vob_id, cell_id, start_sector, last_sector)]

        self.nr_of_vobs = vobcnt
        self.cell_adr_table = vts_c_adt
