import os


class Config:
    all_chroms = list(range(1, 23))
    nr_folds = 5
    folds = [[1, 11, 20, 13], [2, 10, 14, 19, 21], [3, 12, 16, 17, 22], [9, 15, 6, 7], [8, 18, 4, 5]]

    def setup(self, **kwargs):
        genome_dict = kwargs.get('genome', {})
        assert all(p in genome_dict for p in ['fa', 'blacklist', 'unmappable'])
        self.genome = genome_dict['fa']
        self.genome_blacklist = genome_dict['blacklist']
        self.genome_unmappable = genome_dict['unmappable']

        dir_dict = kwargs.get('dir', {})
        assert all(f in dir_dict for f in ['logs', 'checkpoint', 'generated', 'tmp'])
        self.logs = dir_dict['logs']
        self.checkpoint = dir_dict['checkpoint']
        self.generated = dir_dict['generated']
        self.tmp = dir_dict['tmp']

        self.binning_type = kwargs.get('binning_type', 'max')
        self.adjust_offset = kwargs.get('adjust_offset', False)

        self.assure_folders()

    def assure_folders(self):
        folders = [self.checkpoint, self.logs, self.generated, self.tmp]
        for folder in folders:
            if not os.path.exists(folder):
                print('Making new directory', folder)
                os.mkdir(folder)
        print(f'Using folders at {[os.path.abspath(f) for f in folders]}')

    def get_chrom_fold(self, chrom: int):
        for i, fold in enumerate(self.folds):
            if chrom in fold:
                return i
    
    def get_chrom_split(self, fold_id):
        assert fold_id in range(self.nr_folds)
        val_chroms = self.folds[fold_id]
        test_chroms = self.folds[(fold_id+1)%self.nr_folds]
        train_chroms = [x for x in self.all_chroms if (x not in val_chroms) and (x not in test_chroms)]
        return train_chroms, val_chroms, test_chroms
    
    def get_fold(self, chrom):
        for i, chroms in enumerate(self.folds):
            if chrom in chroms:
                return (i+1)%self.nr_folds
        return None


CONFIG = Config()
