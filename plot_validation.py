import re
import matplotlib.pyplot as plt

raw_text = """        Val pearson r: 0.6964387753816834
        Val pearson r: 0.6870994218353818
        Val pearson r: 0.7291391529590228
        Val pearson r: 0.7444582223654782
        Val pearson r: 0.6241408496209283
        Val pearson r: 0.6560767789907609
        Val pearson r: 0.6151724151653086
        Val pearson r: 0.7034725605829238
        Val pearson r: 0.6497986103344944
        Val pearson r: 0.6155472763204469
        Val pearson r: 0.6552607604094044
        Val pearson r: 0.630054931594626
        Val pearson r: 0.6185279754623397
        Val pearson r: 0.7216549256354842
        Val pearson r: 0.7055036708491718
        Val pearson r: 0.7511119742668912
        Val pearson r: 0.7635181946177819
        Val pearson r: 0.6537835362569802
        Val pearson r: 0.6835558226295194
        Val pearson r: 0.6388907881539803
        Val pearson r: 0.7190023912652485
        Val pearson r: 0.6743860834015354
        Val pearson r: 0.6469024054589176
        Val pearson r: 0.6794421250988107
        Val pearson r: 0.6472221303262831
        Val pearson r: 0.640377184784484
        Val pearson r: 0.7295442255629667
        Val pearson r: 0.7136852801487562
        Val pearson r: 0.7589122433393578
        Val pearson r: 0.7707581572647464
        Val pearson r: 0.6622760109618123
        Val pearson r: 0.6936889889396246
        Val pearson r: 0.6511044945298297
        Val pearson r: 0.7263597983993362
        Val pearson r: 0.6833798111148207
        Val pearson r: 0.6560634099565105
        Val pearson r: 0.6966597691502155
        Val pearson r: 0.6540557195808048
        Val pearson r: 0.6567890556048219
        Val pearson r: 0.7355667790321698
        Val pearson r: 0.7181773762583913
        Val pearson r: 0.762306945735433
        Val pearson r: 0.7735634438674696
        Val pearson r: 0.6709062488003141
        Val pearson r: 0.7021438188342437
        Val pearson r: 0.6612441596265453
        Val pearson r: 0.7299128731387278
        Val pearson r: 0.6938203863166237
        Val pearson r: 0.6645958744627727
        Val pearson r: 0.7062905065624483
        Val pearson r: 0.6594368702854894
        Val pearson r: 0.6751533285462077
        Val pearson r: 0.7374728764598621
        Val pearson r: 0.7196044648882977
        Val pearson r: 0.7670996660386562
        Val pearson r: 0.7780080149901397
        Val pearson r: 0.6742898696492363
        Val pearson r: 0.7058934639201252
        Val pearson r: 0.6672949258734164
        Val pearson r: 0.7312364038403028
        Val pearson r: 0.6955165547211278
        Val pearson r: 0.6682926530595027
        Val pearson r: 0.7098759765598139
        Val pearson r: 0.6632259104204266
        Val pearson r: 0.6801936203807368
        Val pearson r: 0.7396180166522474
        Val pearson r: 0.7219697031367717
        Val pearson r: 0.7700792666119827
        Val pearson r: 0.7812709213412251
        Val pearson r: 0.6768472576134353
        Val pearson r: 0.7089772560114744
        Val pearson r: 0.6724248145994438
        Val pearson r: 0.7328345447485672
        Val pearson r: 0.70032733213575
        Val pearson r: 0.6708278062936569
        Val pearson r: 0.712862836202867
        Val pearson r: 0.6696105337131105
        Val pearson r: 0.6866884996048938
        Val pearson r: 0.7411891055219084
        Val pearson r: 0.7245902689326984
        Val pearson r: 0.7708178484437126
        Val pearson r: 0.7812205607572208
        Val pearson r: 0.6805039766603462
        Val pearson r: 0.7134462216944767
        Val pearson r: 0.6765413814193819
        Val pearson r: 0.7341053857499662
        Val pearson r: 0.7045731751043073
        Val pearson r: 0.6743634373727768
        Val pearson r: 0.716458337219881
        Val pearson r: 0.6729699794543645
        Val pearson r: 0.6892055798526996
        Val pearson r: 0.7405559298361586
        Val pearson r: 0.7237466444041802
        Val pearson r: 0.7730430125437816
        Val pearson r: 0.7841607376771775
        Val pearson r: 0.679896613150622
        Val pearson r: 0.7170980883605286
        Val pearson r: 0.679135088372655
        Val pearson r: 0.7333611224807183
        Val pearson r: 0.704946036120285
        Val pearson r: 0.6750229864727569
        Val pearson r: 0.7170640526514216
        Val pearson r: 0.6754801310366083
        Val pearson r: 0.6892097632370298
        Val pearson r: 0.7457581148412622
        Val pearson r: 0.7281806164610638
        Val pearson r: 0.775695809322687
        Val pearson r: 0.7861311517116034
        Val pearson r: 0.6837907068494478
        Val pearson r: 0.7254445193158302
        Val pearson r: 0.6826638561237102
        Val pearson r: 0.7371976392755624
        Val pearson r: 0.7129005819104313
        Val pearson r: 0.6785930010014146
        Val pearson r: 0.7203582832818095
        Val pearson r: 0.6786407513936481
        Val pearson r: 0.6946855095794354
        Val pearson r: 0.7439361343698927
        Val pearson r: 0.7271142408196352
        Val pearson r: 0.7758318372488294
        Val pearson r: 0.7857087072185104
        Val pearson r: 0.6817283255573829
        Val pearson r: 0.72494311855335
        Val pearson r: 0.6845077675512952
        Val pearson r: 0.7362547679616372
        Val pearson r: 0.7108034558528062
        Val pearson r: 0.6770417456291605
        Val pearson r: 0.7195329931669482
        Val pearson r: 0.6786599228741661
        Val pearson r: 0.6946550478293225
        Val pearson r: 0.7412733773670532
        Val pearson r: 0.7245576204965146
        Val pearson r: 0.7746532810950112
        Val pearson r: 0.7848534129186068
        Val pearson r: 0.6793816030317934
        Val pearson r: 0.7236412771844641
        Val pearson r: 0.6818868719419771
        Val pearson r: 0.7336746683379142
        Val pearson r: 0.7099762496923312
        Val pearson r: 0.6744342750812531
        Val pearson r: 0.7184549823385983
        Val pearson r: 0.6760434079584795
        Val pearson r: 0.6925774202111611
        Val pearson r: 0.7424062650662471
        Val pearson r: 0.7266368686876203
        Val pearson r: 0.7755794476134695
        Val pearson r: 0.7842987925516092
        Val pearson r: 0.68165292486155
        Val pearson r: 0.7258095444959102
        Val pearson r: 0.6834873311576279
        Val pearson r: 0.7354560756083379
        Val pearson r: 0.71199184196435
        Val pearson r: 0.6763740993440086
        Val pearson r: 0.7189250457504082
        Val pearson r: 0.6788830698904805
        Val pearson r: 0.6943106983132857
        Val pearson r: 0.7450365912906849
        Val pearson r: 0.7288023282895445
        Val pearson r: 0.7765684548470875
        Val pearson r: 0.7856408276313993
        Val pearson r: 0.6818362862528512
        Val pearson r: 0.7281351907168422
        Val pearson r: 0.6848795112165097
        Val pearson r: 0.736865135898467
        Val pearson r: 0.7149251550514664
        Val pearson r: 0.6773219620250895
        Val pearson r: 0.7227276417064363
        Val pearson r: 0.6809720803882103
        Val pearson r: 0.695572395715137
        Val pearson r: 0.7416012986516195
        Val pearson r: 0.7273640296381129
        Val pearson r: 0.7773683142944958
        Val pearson r: 0.7862939212395638
        Val pearson r: 0.6804398988843954
        Val pearson r: 0.7260217080921356
        Val pearson r: 0.6836087222999384
        Val pearson r: 0.7352362622176543
        Val pearson r: 0.7121195524652735
        Val pearson r: 0.674739671521881
        Val pearson r: 0.7216534767164122
        Val pearson r: 0.679829676185234
        Val pearson r: 0.6943985773305369"""

datasets = ['HCT116', 'A549_RGS', 'WTC11', 'GM23338', 'HG03432', 'MCF7', 'PC3', 'Panc1', 'RWPE2', 'GM12878_XSC', 'HEPG2_GJU', 'K562_FGK', 'IMR90']

# Extract all floating point numbers
values = [float(x) for x in re.findall(r"\d+\.\d+", raw_text)]

heads = 13
epochs = len(values) // heads

# Reshape into (epochs x heads)
data = [values[i*heads:(i+1)*heads] for i in range(epochs)]

# Transpose to get per-head time series
per_head = list(zip(*data))

# Plot
plt.figure()

for i, head_vals in enumerate(per_head):
    plt.plot(range(1, epochs + 1), head_vals, label=f"{datasets[i]}")

plt.xlabel("Epoch")
plt.ylabel("Validation Pearson r")
plt.title("Validation Scores per Head over Epochs")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

plt.savefig("validation_scores_13head.png")