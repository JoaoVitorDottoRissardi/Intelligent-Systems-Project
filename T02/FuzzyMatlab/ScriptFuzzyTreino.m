    
    clear ; close all; clc
    prompt = 'Digite o nome do arquivo de regras Fuzzy que deseja utilizar\n';
    escolhaRegras = input(prompt, 's');
    
    opts = detectImportOptions('treino_sinais_vitais_com_label.txt');     
    preview('treino_sinais_vitais_com_label.txt',opts);
    opts.SelectedVariableNames = [4 5 6];
    fuzzy_inputs = readmatrix('treino_sinais_vitais_com_label.txt', opts);
    opts.SelectedVariableNames = [7];
    fuzzy_outputs = readmatrix('treino_sinais_vitais_com_label.txt', opts);
    opts.SelectedVariableNames = [8];
    fuzzy_outputsCat = readmatrix('treino_sinais_vitais_com_label.txt', opts);

    fis = readfis(escolhaRegras);

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

    contagem = 0;
    comparacao = 0;
    resultados2 = zeros(numDados, numOutputs);

    for j = 1:numDados
        if(resultados(j)<25 && resultados(j)>0)
            resultados2(j) =1;
        elseif(resultados(j)<50)
            resultados2(j) =2;
        elseif(resultados(j)<75)
            resultados2(j) =3;       
        elseif(resultados(j)<100)
            resultados2(j) =4;
        end
        if(resultados2(j)==fuzzy_outputsCat(j))
            contagem = contagem + 1;
        end   
    end
    acerto = contagem*100/numDados;

    % Create a table with the data and variable names
    T = table(resultados2);
    % Write data to text file
    writetable(T, 'ResultadosFuzzyTreino.txt', 'WriteVariableNames',false);

    disp(acerto);
    disp("Os resultados foram gerados no arquivo: ResultadosFuzzyTreino.txt");
