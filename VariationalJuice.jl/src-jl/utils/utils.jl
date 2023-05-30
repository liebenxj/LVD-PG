using CUDA


function select_gpu(idx)
    device!(collect(devices())[idx+1])
end

function overprint(str)  
    print("\u1b[1F")
    #Moves cursor to beginning of the line n (default 1) lines up   
    print(str)   #prints the new line
    print("\u1b[0K") 

    println() #prints a new line, i really don't like this arcane codes
end