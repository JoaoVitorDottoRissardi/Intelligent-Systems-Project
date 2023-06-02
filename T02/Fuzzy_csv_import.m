clear ; close all; clc

opts = detectImportOptions('treino_sinais_vitais_com_label.txt');
preview('treino_sinais_vitais_com_label.txt',opts)
opts.SelectedVariableNames = [4 5 6];
fuzzy_inputs = readmatrix('treino_sinais_vitais_com_label.txt', opts);
opts.SelectedVariableNames = [7];
fuzzy_outputs = readmatrix('treino_sinais_vitais_com_label.txt', opts);