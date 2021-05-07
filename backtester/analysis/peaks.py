# -*- coding: utf-8 -*-

def peak_locator(x, minrange):
    last_maxima = x[0]
    current_delta = x[0]
    current_loc = 0
    
    xlen = len(x)
    
    jj = 0 
    for ii in range(xlen):
        xi = x[ii]
        delta = xi - last_maxima
        abs_delta = abs(delta)
        if abs_delta > current_delta:
            current_delta = abs_delta
        elif 
        
    
    
    

@njit
def trough_volume(x: np.ndarray):
    current_max = x[0]
    current_max_loc = 0
    xlen = len(x)
    areas = np.zeros(xlen)
    
    # Record some information about each trough
    start_locs = np.zeros(xlen, dtype=np.int64)
    end_locs = np.zeros(xlen, dtype=np.int64)
    # min_locs = np.zeros(xlen, dtype=np.int64)
    areas_final = np.zeros(xlen)
    
    area = 0
    jj = 0
    for ii in range(xlen):
        xi = x[ii]
        
        # Price is rising
        if xi > current_max:
            start_locs[jj] = current_max_loc
            end_locs[jj] = ii
            areas_final[jj] = area
            jj += 1
            area = 0
            current_max_loc = ii
            current_max = xi
        # Trough detected
        else:
            area += (current_max - xi)
        
        areas[ii] = area    
        
    # If there's are left over record it for the final trough. 
    if area > 0:
        start_locs[jj] = current_max_loc
        end_locs[jj] = ii
        areas_final[jj] = area
        jj += 1
        
    start_locs = start_locs[0 : jj]
    end_locs = end_locs[0 : jj]
    areas_final = areas_final[0 : jj]
    return areas, start_locs, end_locs, areas_final