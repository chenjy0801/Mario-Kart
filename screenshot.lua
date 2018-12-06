RECORDING_FOLDER = 'screenshot'
local recording_frame = 1

while (True) do
	print("shot")
	
	client.screenshot(RECORDING_FOLDER .. '\\' .. recording_frame .. '.png')
	recording_frame = recording_frame +1

	for i = 50, 1, -1 do
		print(i)
	end

end 