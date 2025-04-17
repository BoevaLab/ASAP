class CellLineConfig:
    def __init__(self, cell_line: str, exp_name: str, bw: str, peaks: str, blacklist: str, lower_bound: int, snv: str,
                 bam: str):
        self.cell_line = cell_line
        self.exp_name = exp_name
        self.bw = bw
        self.peaks = peaks
        self.blacklist = blacklist
        self.lower_bound = lower_bound
        self.snv = snv
        self.bam = bam

    def __repr__(self):
        return self.__dict__.__str__()
