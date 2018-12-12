local util = {}

local uuid = require("lualibs.uuid"); uuid.seed()
function util.generateUUID()
  return uuid()
end

function util.getWorkingDir()
  return io.popen("cd"):read("*l")
end

-- Return the location of the TMP dir on this computer, caching the result.
local TMP_DIR = nil
function util.getTMPDir()
  if TMP_DIR == nil then TMP_DIR = io.popen("echo %TEMP%"):read("*l") end
  return TMP_DIR
end

-- Sign, Clamp, and Lerp functions, taken from lume
function util.sign(x) return x < 0 and -1 or 1 end
function util.clamp(x, min, max) return x < min and min or (x > max and max or x) end
function util.lerp(a, b, amount) return a + (b - a) * util.clamp(amount, 0, 1) end

function util.linspace(start, vend, divs)
  local val = start
  local step_size = (vend - start) / (divs - 1)
  local i = -1
  return function ()
    i = i + 1
    if i < divs - 1 then
      return start + i * step_size
    elseif i == divs - 1 then
      return vend
    end
  end
end

function util.finiteDifferences(nums)
  local diffs = {}
  for i, x in ipairs(nums) do
    if i > 1 then table.insert(diffs, x - nums[i - 1]) end
  end
  return diffs
end

function util.bendingEnergy(nums)
  local accum = 0
  local second_deriv = util.finiteDifferences(util.finiteDifferences(nums))
  for _, x in ipairs(second_deriv) do accum = accum + x * x end
  return accum
end

function util.readPlayerX() return mainmemory.readfloat(0x0F69A4, true) end
function util.readPlayerY() return mainmemory.readfloat(0x0F69AC, true) end
function util.readPlayerZ() return mainmemory.readfloat(0xF69A8, true) end
function util.readPlayerPos()
  return {util.readPlayerX(), util.readPlayerY(), util.readPlayerZ()}
end

-- =============================================================== --

--util.CHARACTER = 0x0DC53B
--function util.readCharacter()
--  return mainmemory.read_u8(util.CHARACTER)
--end

util.ITEM = 0x165F5B
function util.readItem()
	return memory.read_s8(util.ITEM)
end


util.LastItemAddr = 0x1658FD
function util.lastitem()
	return mainmemory.read_u8(util.LastItemAddr)
end

util.itemtable=0x802A7BD4
function util.itemtable()
	return mainmemory.readbyte(util.itemtable)
end





util.StateCAddr = 0x0F6A4C
function util.C()
	return mainmemory.read_u8(util.StateCAddr)
end

util.StateEAddr = 0x0F6A4E
function util.E()
	return mainmemory.read_u8(util.StateEAddr)
end
    

util.StateFAddr = 0x0F6A4F
function util.F()
	return mainmemory.read_u8(util.StateFAddr)
end
 
util.State5BAddr = 0x0F6A5B
function util.B()
    return mainmemory.read_u8(util.State5BAddr)
end

util.MTglideAddr = 0x0F6BCB
function util.mt()
	return mainmemory.read_u8(util.MTglideAddr)
end





util.AA = 0x0F699C
function util.AAA()
	return mainmemory.read_u8(util.AA)
end

util.AB = 0x0F699D
function util.AAB()
	return mainmemory.read_u8(util.AB)
end

util.AC = 0x0F699E
function util.AC()
	mainmemory.write_u8(0x0F699E, 32)
end
function util.DC()
	mainmemory.write_u8(0x0F699E, 0)
end

function util.shrooming()
  mainmemory.write_u8(0x0F6A4E, 32)
end

util.AD = 0x0F699F
function util.AAD()
	return mainmemory.read_u8(util.AD)
end

function util.slide()
  mainmemory.write_u8(0x0F6A4C,32)
end

-- =============================================================== --




-- Read the current progress in the course from memory.
util.PROGRESS_ADDRESS = 0x1644D0
function util.readProgress()
  return mainmemory.readfloat(util.PROGRESS_ADDRESS, true)
end

-- Read the velocity of the player from meory.
util.VELOCITY_ADDRESS = 0x0F6BBC
function util.readVelocity()
  return mainmemory.readfloat(util.VELOCITY_ADDRESS, true)
end

-- The current match timer.
util.TIMER_ADDRESS = 0x0DC598
function util.readTimer()
  return mainmemory.readfloat(util.TIMER_ADDRESS, true)
end

-- The current mode.
util.MODES = {"GP", "TT", "VS", "BT"}
util.MODE_NAMES = {"Grand Prix", "Time Trial", "VS.", "Battle"}
util.MODE_ADDRESS = 0x0DC53F
function util.readMode()
  local i = mainmemory.read_u8(util.MODE_ADDRESS)
  return util.MODES[i + 1]
end

util.COURSES = {"MR","CM","BC","BB","YV","FS","KTB","RRy","LR","MMF","TT","KD","SL","RRd","WS",
  "BF","SS","DD","DK","BD","TC"}
util.COURSE_NAMES = {"Mario Raceway", "Choco Mountain", "Bowser's Castle", "Banshee Boardwalk",
  "Yoshi Valley", "Frappe Snowland", "Koopa Troopa Beach", "Royal Raceway", "Luigi Raceway",
  "Moo Moo Farm", "Toad's Turnpike", "Kalimari Desert", "Sherbert Land", "Rainbow Road",
  "Wario Raceway", "Block Fort", "Skyscraper", "Double Deck", "D.K.'s Jungle Parkway", "Big Donut", "TC"}


--===============================================--
--util.actions={["0"]=0,["-0.4"]=1,["-0.2"]=2,["0.2"]=3,["0.4"]=4,["-"]=5,["0.2"]=6,["0.4"]=7,["0.6"]=8,["0.8"]=9,["1"]=10}
--util.reactions={0,-0.4,-0.2,0.2,0.4,-1,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1}
util.actions = {["-1"]=5, ["-0.8"]=6, ["-0.6"]=7, ["-0.4"]=8, ["-0.2"]=9, ["0.2"]=10, ["0.4"]=11, ["0.6"]=12, ["0.8"]=13, ["1"]=14}
util.actions1 = {["0"]=0, ["-0.4"]=1, ["-0.2"]=2,["0.2"]=3,["0.4"]=4}

util.reactions = {-1,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1}
util.reactions1 = {-0.4,-0.2,0.2,0.4}
--    drift  steer
-- 0 :  0      0 
-- 1 :  0     -0.4 
-- 2 :  0     -0.2 
-- 3 :  0      0.2
-- 4 :  0      0.4 
-- 5:   1      -1 
-- 6:   1      -0.8
-- 7:   1      -0.6
-- 8:   1      -0.4 
-- 9:   1      -0.2
-- 10:  1       0.2
-- 11:  1       0.4
-- 12:  1       0.6
-- 13:  1       0.8
-- 14:  1       1

--===============================================--



util.COURSE_ADDRESS = 0x0DC5A1
function util.readCourse()
  local i = mainmemory.read_u8(util.COURSE_ADDRESS)
  return util.COURSES[i + 1]
end


util.STEER_MIN, util.STEER_MAX = -1, 1
util.JOYSTICK_MIN, util.JOYSTICK_MAX = -128, 127

function util.convertSteerToJoystick(steer, use_mapping)
  -- Ensure that steer is between -1 and 1
  steer = util.clamp(steer, util.STEER_MIN, util.STEER_MAX)

  -- If we are using our mapping, map the linaer steer space to the joystick space.
  if use_mapping == true then
    steer = util.sign(steer) * math.sqrt(math.abs(steer) * 0.24 + 0.01)
  end

  -- Map the -1 to 1 steer into an integer -128 to 127 space.
  local alpha = (steer + 1)/2
  return math.floor(util.lerp(util.JOYSTICK_MIN, util.JOYSTICK_MAX, alpha))
end

function util.split(delim,str)

    local t = {}

    for substr in string.gmatch(str, "[^".. delim.. "]*") do

        if substr ~= nil and string.len(substr) > 0 then

            table.insert(t,substr)

        end

    end

    return t

end

return util
