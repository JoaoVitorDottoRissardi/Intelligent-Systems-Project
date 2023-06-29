    
    clear ; close all; clc
    escolha = input("Digite 1 para Defusificação centroid \nDigite 2 para Defusificação bisector \n");
    opts = detectImportOptions('../treino_sinais_vitais_com_label.txt');
    preview('../treino_sinais_vitais_com_label.txt',opts)
    opts.SelectedVariableNames = [4 5 6];
    fuzzy_inputs = readmatrix('../treino_sinais_vitais_com_label.txt', opts);
    opts.SelectedVariableNames = [7];
    fuzzy_outputs = readmatrix('../treino_sinais_vitais_com_label.txt', opts);
    opts.SelectedVariableNames = [8];
    fuzzy_outputsCat = readmatrix('../treino_sinais_vitais_com_label.txt', opts);
    
    % Carregue o sistema fuzzy a partir do arquivo FIS (Fuzzy Inference System) criado
    if(escolha == 1)
        fis = readfis('FuzzyOficial1.fis');
    elseif(escolha == 2)
        fis = readfis('Fuzzy10-bisector.fis');
    end
    % Obtenha o número de saídas do sistema fuzzy
    numOutputs = numel(fis.output);
    % Crie uma matriz para armazenar os resultados das saídas
    numDados = size(fuzzy_inputs, 1);
    resultados = zeros(numDados, numOutputs);
    % Itere sobre os dados
    for i = 1:numDados
        % Armazene os resultados das saídas
        resultados(i, :) = evalfis(fuzzy_inputs(i, 1:3), fis);
    end
    % Exiba os resultados
    % disp(resultados);

    comparacao = 0;

    for j = 1:numDados
        auxCat=0;
        if(resultados(j)<25 && resultados(j)>0)
            auxCat=1;
        elseif(resultados(j)<50)
            auxCat=2;
        elseif(resultados(j)<75)
            auxCat=3;        
        elseif(resultados(j)<100)
            auxCat=4;
        end
        if (auxCat == fuzzy_outputsCat(j))
            comparacao = comparacao + 1;
        end
    end

    acerto = 100*comparacao/numDados;
    clc
    disp("Acerto: " + acerto + "%");

