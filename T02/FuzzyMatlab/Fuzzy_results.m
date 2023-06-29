%fis = readfis('Fuzzy7-simples.fis');

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
disp(resultados);