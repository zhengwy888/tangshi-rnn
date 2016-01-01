require 'lfs'
local YunjiaoLoader = {}
YunjiaoLoader.__index = YunjiaoLoader
-- data: http://baike.baidu.com/view/584090.htm?fromtitle=%E4%B8%83%E5%BE%8B&fromid=96270&type=syn#2_2
-- 平水韵: https://zh.wikisource.org/zh-hans/%E5%B9%B3%E6%B0%B4%E9%9F%BB
-- 0: whatever
-- 1: 平
-- 2: 仄
-- 3: 平韵脚 （近体诗只押平声韵）
-- 9: , .
local orig_rules = {
-- 七绝
{'0102213',
'0211223',
'0201122',
'0102213'},

{'0102112',
'0211223',
'0201122',
'0102213'},

{'0211223',
'0102213',
'0102112',
'0211223'},

{'0201122',
'0102213',
'0102112',
'0211223'},

-- 七律
{
'0102211','0211223',
'0201122','0102213',
'0102112','0211223',
'0201122','0102213',
},
{
'0102112','0211223',
'0201122','0102213',
'0102112','0211223',
'0201122','0102213',
},
{
'0211221','0102213',
'0102112','0211223',
'0201122','0102213',
'0102112','0211223',
},
{
'0201122','0102213',
'0102112','0211223',
'0201122','0102213',
'0102112','0211223',
},

-- 五律
{'02211','11223',
'01122','02213',
'02112','11223',
'01122','02213',
},{
'02112','11223',
'01122','02213',
'02112','11223',
'01122','02213',
},{
'11221','02213',
'02112','11223',
'01122','02213',
'02112','11223',
},{
'01122','02213',
'02112','11223',
'01122','02213',
'02112','11223',
},
-- 五绝
{'02112','11223','01122','02213',
},{
'02211','11223','01122','02213',
},{
'01122','02213','02112','11223',
},{
'11221','02213','02112','11223',
}
}

local rules = {}
for _, rule in ipairs(orig_rules) do
    table.insert(rules, table.concat(rule, '9'))
end

-- charToYun: 翩: {1, 20}
-- yunToChar: {1 = {20={'翩'}}
function YunjiaoLoader.create()
    local self = {}
    setmetatable(self, YunjiaoLoader)

    -- th contains path package, but it breaks when only uses luajit
    -- local ze_file = path.join(path.dirname(paths.thisfile()), 'ze.txt');
    -- local ping_file = path.join(path.dirname(paths.thisfile()), 'ping.txt');
    -- local yunjiao_file = path.join(path.dirname(paths.thisfile()), 'yunjiao.t7')
    local ze_file = paths.dirname(paths.thisfile()) .. '/ze.txt'
    local ping_file = paths.dirname(paths.thisfile()).. '/ping.txt'
    local yunjiao_file = paths.dirname(paths.thisfile()).. '/yunjiao.t7'

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if not file_exists(yunjiao_file) then
        -- prepro files do not exist, generate them
        print('yunjiao.t7 do not exist. Running preprocessing...')
        run_prepro = true
    else
        -- check if the input file was modified since last time we 
        -- ran the prepro. if so, we have to rerun the preprocessing
        local input_attr = lfs.attributes(ping_file)
        local vocab_attr = lfs.attributes(yunjiao_file)
        local ze_attr = lfs.attributes(ze_file)
        if input_attr.modification > vocab_attr.modification 
            or ze_attr.modification > vocab_attr.modification then
            print('yunjiao.t7 detected as stale. Re-running preprocessing...')
            run_prepro = true
        end
    end
    if run_prepro then
        -- construct a tensor with all the data, and vocab file
        print('one-time setup: preprocessing input text file ...')
        YunjiaoLoader.text_to_yunjiao(ping_file, ze_file, yunjiao_file)
    end

    self.mapping = torch.load(yunjiao_file)

    collectgarbage()
    return self
end

function YunjiaoLoader:getYunjiao(char)
    if not self.mapping['charToYun'][char] then
        error("out of range char " .. char)
    end
    local idx = self.mapping['charToYun'][char]
    return self.mapping['yunToChar'][idx[1]][idx[2]]
end

function YunjiaoLoader:getPingze(char)
    if not self.mapping['charToYun'][char] then
        error("out of range char " .. char)
    end
    return self.mapping['charToYun'][char][1]
end

function YunjiaoLoader:getPingzeVocab(pingze)
    return flatten(self.mapping['yunToChar'][pingze])
end

-- returns nil if there is no possible extension
-- returns 9 if sentence needs to be finished
function YunjiaoLoader:getNextPingzes(poem)
    if #poem == 0 then
        return {0}
    else 
        local poemPingze = {}
        for char in string.gfind(poem, "([%z\1-\127\194-\244][\128-\191]*)") do
            table.insert(poemPingze, self:getPingze(char))
        end
        local validPingze = {}
        for _, rule in ipairs(rules) do
            if YunjiaoLoader.matchRule(rule, poemPingze) then
                local n = #poemPingze+1
                validPingze[string.sub(rule,n,n)] = true
            end
        end
        local pingzeList = {}
        for key, _ in pairs(validPingze) do
            table.insert(pingzeList, key)
        end
        return pingzeList
    end
end

function YunjiaoLoader:belongsTo(pingzeList, char) 
    local pingze = tonumber(self:getPingze(char))
    for _, _p in pairs(pingzeList) do
        p = tonumber(_p)
        if p == 0 and pingze ~= 9 then
            return true
        end
        if p == 3 then
            p = 1
        end
        if pingze == p then
            return true
        end
    end
    return false
end

-- *** STATIC method ***
function YunjiaoLoader.text_to_yunjiao(ping_infile, ze_infile, out_yunjiaofile)
    local data = {
        charToYun= {},
        yunToChar= { {}, {} },
    }

    local rawdata ={}
    local f = torch.DiskFile(ping_infile)
    table.insert(rawdata, f:readString('*a')) -- NOTE: this reads the whole file at once
    f:close()
    local f = torch.DiskFile(ze_infile)
    table.insert(rawdata, f:readString('*a')) -- NOTE: this reads the whole file at once
    f:close()
    for pingze, content in ipairs(rawdata) do
        local yunIdx = 1
        for char in string.gfind(content, "([%z\1-\127\194-\244][\128-\191]*)") do
            if char == "\n" then
                yunIdx = yunIdx + 1
            else
                if not data['yunToChar'][pingze][yunIdx] then data['yunToChar'][pingze][yunIdx] = {} end
                table.insert(data['yunToChar'][pingze][yunIdx], char)
                data['charToYun'][char] = {pingze,yunIdx}
            end
        end
    end

    data['charToYun']["，"] = {9,1}
    data['charToYun']["。"] = {9,2}

    torch.save(out_yunjiaofile, data)
    return data
end

function YunjiaoLoader.matchRule(rule, pingzeList)
    if #pingzeList > #rule then
        return false
    end
    for i, p in ipairs(pingzeList) do
        local r = string.sub(rule, i, i)
        if r == '3' and tostring(p) == '1' then
        
        elseif r ~= '0' and r ~= tostring(p) then
            return false
        end
    end
    return true
end


function flatten(arr)
    local result = { }

    local function flatten(arr)
        for _, v in ipairs(arr) do
            if type(v) == "table" then
                flatten(v)
            else
                table.insert(result, v)
            end
        end
    end

    flatten(arr)
    return result
end

function unicodeLength(ustring) 
    local count = 0
    for char in string.gfind(content, "([%z\1-\127\194-\244][\128-\191]*)") do
        count = count + 1
    end
    return count
end

function startWith(String,Start)
   return string.sub(String,1,string.len(Start))==Start
end

function file_exists(name)
    if type(name)~="string" then return false end
    return os.rename(name,name) and true or false
end

return YunjiaoLoader




