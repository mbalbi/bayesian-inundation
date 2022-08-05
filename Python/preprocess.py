"""

Funciton to write Lisflood-fp input files

"""
import time

def wait_timeout(proc, seconds):
    """
    """
    start = time.time()
    end = start + seconds
    interval = min(seconds/1, .25)

    while True:
        result = proc.poll()
        if result is not None:
            return result
        if time.time() >= end:
            proc.kill()
            return -1
        time.sleep(interval)

def create_river( filename, nodes, width, r_ch, bed_i, bed_f, bci_key_i,
                  bci_val_i, bci_key_f=None, bci_val_f=None,  ntribs=1 ):
    
    # Create and write file
    river_file = open(filename,'w')
    
    # Tribs
    river_file.write( 'Tribs ' + str(ntribs) + '\n' )
    
    # Number of river nodes
    river_file.write( str(nodes.shape[0]) + '\n' )
    
    # First row of nodes
    river_file.write( str(nodes[0,0]) + '\t' + str(nodes[0,1]) + '\t' + \
                      str(width) + '\t' + str(r_ch) + '\t' +  str(bed_i) + \
                      '\t' + bci_key_i + '\t' + str(bci_val_i) + '\n' )
    
    # Middle rows of nodes
    for i in range(1, nodes.shape[0]-1):
        river_file.write( str(nodes[i,0]) + '\t' + str(nodes[i,1])+ '\n' )
    
    # Last node
    river_file.write( str(nodes[-1,0]) + '\t' + str(nodes[-1,1]) + '\t' + \
                      str(width) + '\t' + str(r_ch) + '\t' + str(bed_f) )
    if bci_key_f:
        river_file.write( '\t' + bci_key_f + '\t' + str(bci_val_f) )
    
    # Last empty line
    river_file.write( '\n' )
    
    # Close file
    river_file.close()

def create_par(filename, files, title='', settings=[], **kwargs):
    """
    """
    # Create and write file
    par_file = open(filename,'w')

    # Title row
    par_file.write('# '+title+'\n')
    par_file.write('\n')

    # Optional key-word arguments
    for key, value in kwargs.items():
        par_file.write(key+"\t"+str(value)+'\n')

    # Files
    for key, value in files.items():
        par_file.write(key+"\t"+str(value)+'\n')
        
    # Single-word settings
    par_file.write('\n')
    for setting in settings:
        par_file.write(setting+'\n')

    # Close file
    par_file.close()

def modify_bci(filename, bc, mod):
    """
    """
    # read bci file
    readFile = open(filename,'r')
    lines = readFile.readlines()
    new_lines = []
    # Go to indicated line
    for line in lines:
        lsplit = line.split()
        if lsplit[0]==bc:
            deleted_line = line
            continue
        new_lines.append(line)
    readFile.close()
    # Write again with last line modified
    w = open(filename,'w')
    ll = deleted_line.split()
    ll[-2] = mod[0]
    ll[-1] = mod[1]
    ll += ['\n']
    new_lines = ['\t'.join(ll)] + new_lines
    w.writelines(new_lines)
    w.close()

def create_bci(filename, bci):
    """
    """
    # Create and write file
    bci_file = open(filename,'w')
    # Boundary conditions
    for bc in bci:
        bci_file.write(bc[0]+'\t'+bc[1]+'\t'+bc[2]+'\t'+bc[3]+'\t'+bc[4]+'\n')
    # Close file
    bci_file.close()

def create_bdy(filename, title, name, q_series):
    """

    q_series [dict]: {'time':[...], 'q':[...]}
    """
    # Create and write file
    bdy_file = open(filename,'w')
    # Title row
    bdy_file.write('# '+title+'\n')
    # Discharge series name
    bdy_file.write(name+'\n')
    #
    time = q_series["time"]
    q = q_series["q"]
    bdy_file.write("\t"+str(len(time))+"\t"+"seconds"+"\n")
    # Discharge time series
    for i in range(len(time)):
        bdy_file.write("\t"+str(q[i])+"\t"+str(time[i])+'\n')
    # Close file
    bdy_file.close()

##
if __name__ == '__main__':
    pass


