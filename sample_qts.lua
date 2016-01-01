
--[[
th sample.lua -length 20 -primetext '月' -temperature 1 -verbose 0 cv/qts_jinti1/lm_jinti_25_epoch17.50_3.0796.t7 >> moon_sample.txt
This file samples characters from a trained model

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',20,'number of characters to sample')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then gprint('package cunn not found!') end
    if not ok2 then gprint('package cutorch not found!') end
    if ok and ok2 then
        gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end
torch.manualSeed(opt.seed)

local Yunjiao = require 'yunjiao.YunjiaoLoader'
local yunjiao = Yunjiao.create()

-- these characters gets removed during sampling
local exclusion = {'*', '□'}
-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- initialize the rnn state to all zeros
gprint('creating an LSTM with layer ...' .. checkpoint.opt.num_layers)
local current_state
local num_layers = checkpoint.opt.num_layers
current_state = {}
for L = 1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, checkpoint.opt.rnn_size)
    if opt.gpuid >= 0 then h_init = h_init:cuda() end
    table.insert(current_state, h_init:clone())
    table.insert(current_state, h_init:clone())
end
state_size = #current_state

-- do a few seeded timesteps
local seed_text = '^' .. opt.primetext
for c in string.gfind(seed_text, "([%z\1-\127\194-\244][\128-\191]*)") do
    if vocab[c] == nil then
        print(c .. ' is not in the dictionary, replace with generic character')
    end
end

-- start sampling/argmaxing
local topprint = 20
local probs
local poemCount = 0
for k=1, opt.length*2 do
    -- max length for a poem is 64
    local furtherexclusion = {}
    local current_state = {}
    for L = 1,checkpoint.opt.num_layers do
        -- c and h for all layers
        local h_init = torch.zeros(1, checkpoint.opt.rnn_size)
        if opt.gpuid >= 0 then h_init = h_init:cuda() end
        table.insert(current_state, h_init:clone())
        table.insert(current_state, h_init:clone())
    end

    -- always seeding with ^ at least
    if string.len(seed_text) > 0 then
        gprint('seeding with ' .. seed_text)
        gprint('--------------------------')
        for c in string.gfind(seed_text, "([%z\1-\127\194-\244][\128-\191]*)") do
            if vocab[c] == nil then
                c = '*'
            end
            prev_char = torch.FloatTensor{vocab[c]}
            if opt.gpuid >= 0 then prev_char = prev_char:cuda() end
            local lst = protos.rnn:forward{prev_char, unpack(current_state)}
            -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
            current_state = {}
            for i=1,state_size do table.insert(current_state, lst[i]) end
            prediction = lst[#lst] -- last element holds the log probabilities
        end
    else
        -- fill with uniform probabilities over characters (? hmm)
        gprint('missing seed text, using uniform probability over first character')
        gprint('--------------------------')
        prediction = torch.Tensor(1, #ivocab):fill(1)/(#ivocab)
        if opt.gpuid >= 0 then prediction = prediction:cuda() end
    end
    local poem = opt.primetext

    local looprunning = true
    for i=1, 64 do
        if not looprunning then
            break
        end
        -- log probabilities from the previous timestep
        local expectedPingze = yunjiao:getNextPingzes(poem)
        if #expectedPingze == 0 then
            error(poem .. ' is not possible to continue')
        end

        if opt.sample == 0 then
            -- use argmax
            local _, prev_char_ = prediction:max(2)
            prev_char = prev_char_:resize(1)
            probs = torch.exp(prediction):squeeze()
            probs:div(torch.sum(probs)) -- renormalize so probs sum to one
        else
            -- previous present characters
            for _, c in ipairs(furtherexclusion) do
                -- prediction[1][vocab[c]] = -100
                prediction[1][vocab[c]] = prediction[1][vocab[c]] * 2
            end
            -- forbidden characters
            for _, c in ipairs(exclusion) do
                prediction[1][vocab[c]] = -100
            end
            -- use sampling
            prediction:div(opt.temperature) -- scale by temperature
            probs = torch.exp(prediction):squeeze()
            probs:div(torch.sum(probs)) -- renormalize so probs sum to one

            for m = 1, 20 do
                prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
                local tmp_char = ivocab[prev_char[1]]
                if tmp_char == '$' then
                    break
                elseif yunjiao:belongsTo(expectedPingze, tmp_char ) then
                    break
                else
                    if opt.verbose == 1 then
                        print(tmp_char .. " does not match pingze")
                    end
                end
                if m == 20 then
                    if opt.verbose == 1 then
                        print(poem .. " does not fit pingze")
                    end
                    looprunning = false
                    break
                end
            end
        end

        -- already picked, good to go
        local real_char = ivocab[prev_char[1]]
        poem = poem .. real_char
        --[[
        if opt.verbose == 1 or real_char ~= '$' then
            io.write(real_char)
        end]]--
        if real_char ~= "，" and real_char ~= "。" then
            table.insert(furtherexclusion, real_char)
        end
        if opt.verbose == 1 then
            -- calculate the top 10 characters
            local prediction_sorted, sorted_char = torch.sort(probs, 1, true)
            for j = 1, topprint do
                io.write(' ')
                io.write(ivocab[sorted_char[j]])
                io.write('/')
                io.write(string.format("%.2f", probs[sorted_char[j]]))
            end
            io.write("\n")
        end
        if real_char == '$' then
            break
        end

        -- forward the rnn for next character
        local lst = protos.rnn:forward{prev_char, unpack(current_state)}
        current_state = {}
        for i=1,state_size do table.insert(current_state, lst[i]) end
        prediction = lst[#lst] -- last element holds the log probabilities
    end
    if string.sub(poem,#poem, #poem) == '$' then
        io.write(string.sub(poem,0, #poem-1) .. '。')
        poemCount = poemCount +1
        io.write('\n') io.flush()
        if poemCount == opt.length then
            break
        end
    end
end

