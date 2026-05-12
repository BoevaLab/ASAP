T_cell_f_21_lp = [0.681,  0.675, 0.666, 0.656] #0.669
T_cell_f_21_separate = [0.7057,  0.701, 0.699,  0.692] #0.699

a_B_cell_m_22_treated_lp = [0.723, 0.712, 0.706, 0.730] #0.71775
a_B_cell_m_22_treated_separated = [0.702, 0.706, 0.703, 0.707]  #0.704

# request rtx4090
# change order and rerun 19heads
# lp on all new heads
# record runtimes
# keep finetuning 13 to 14


print(sum(T_cell_f_21_lp)/4)
print(sum(T_cell_f_21_separate)/4)

print(sum(a_B_cell_m_22_treated_lp)/4)
print(sum(a_B_cell_m_22_treated_separated)/4)
