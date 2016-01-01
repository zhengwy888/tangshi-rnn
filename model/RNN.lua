local RNN = {}

function RNN.rnn(input_size, rnn_size, n, dropout)
  dropout = dropout or 0 
  
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  -- do I really want to assume this code works? the prev_h and the x seems to be flipped?
  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    local prev_h = inputs[L+1]
    if L == 1 then 
      x = OneHot(input_size)(inputs[1])
      input_size_L = input_size
    else 
      x = outputs[L-1] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- RNN tick
    local i2h = nn.Linear(input_size_L, rnn_size)(x)
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
    local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h})

    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h)
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

function nn.Recurrent:forgetEvalute()
    self.step = 1
    for i,_ in pairs(self.inputs) do
        self.inputs[i] = nil
        self.gradOutputs[i] = nil
    end
    for i, _ in pairs(self.gradOutputs) do
        self.gradOutputs[i] = nil
    end
    for i, _ in pairs(self.sharedClones) do 
        self.sharedClones[i] = nil
    end
    local count=0
    local ta = {}
    --[[for i, _ in pairs(self.outputs) do 
        table.insert(ta, i)
        self.outputs[i] = nil
        count = count+1
    end]]--
    self.outputs = {}
end

function RNN.errnn(input_size, rnn_size,n,rho, dropout)
    -- TODO feedback function can contain dropout, namely nn.linear
    rho = rho or 8
    dropout = dropout or 0
    local hiddenSize = rnn_size
    local nIndex = input_size
    local feedback = nn.Linear(hiddenSize,hiddenSize)
    if dropout > 0 then
        local dfeed = nn.Sequential()
        dfeed:add(feedback)
        dfeed:add(nn.Dropout(dropout))
        feedback = dfeed
    end
    --[[local r = nn.Recurrent(
        hiddenSize, nn.LookupTable(nIndex, hiddenSize), 
        nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
        rho
    )]]--
    local r = nn.Recurrent(
        hiddenSize, nn.LookupTable(nIndex, hiddenSize), 
        feedback, nn.ReLU(), 
        rho
    )
    rnn = nn.Sequential()
    rnn:add(r)
    if dropout > 0 then
        rnn:add(nn.Dropout(dropout))
    end
    rnn:add(nn.Linear(hiddenSize, nIndex))
    rnn:add(nn.LogSoftMax())
    return rnn,r
end

return RNN
