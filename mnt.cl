
__kernel
void compute_given_flowdir(__global float* dem, const int width, const int height, const int nodata, volatile __global int* flowDir, volatile __global int* flowAcc, const int i, const int j){
    if(flowDir[i * width + j] != -1){ // already computed
        return;
    }
    if (dem[i * width + j] == nodata) {
        flowDir[i * width + j] = nodata;
        flowAcc[i * width + j] = nodata;
    } else {
        flowAcc[i * width + j] = 0;
        // compute the min between self and the 8 neighbors
        float min = dem[i * width + j];
        int dir = 0;
        for (int k = -1; k < 2; k++) {
            for (int l = -1; l < 2; l++) {
                if((k==0 && l==0)|| dem[(i + k) * width + j + l] == nodata){
                    continue;
                }
                if (i + k >= 0 && i + k < width && j + l >= 0 && j + l < height) {
                    if (dem[(i + k) * width + j + l] < min) {
                        min = dem[(i + k) * width + j + l];
                        // NO = 1, N = 2 , NE = 3, E = 4, SE = 5, S = 6, SW = 7, W = 8
                        if (k == -1 && l == -1) {
                            dir = 1;
                        } else if (k == -1 && l == 0) {
                            dir = 2;
                        } else if (k == -1 && l == 1) {
                            dir = 3;
                        } else if (k == 0 && l == 1) {
                            dir = 4;
                        } else if (k == 1 && l == 1) {
                            dir = 5;
                        } else if (k == 1 && l == 0) {
                            dir = 6;
                        } else if (k == 1 && l == -1) {
                            dir = 7;
                        } else if (k == 0 && l == -1) {
                            dir = 8;
                        }
                    }
                }
            }
        }
        flowDir[i * width + j] = dir;
    }
}

__kernel
void acc_recursize(volatile __global int* flowDir,volatile __global int* flowAcc, const int i, const int j, const int width, const int height, const int nodata, __global float* dem){
    if (i < 0 || i >= height || j < 0 || j >= width) {
        return;
    }
    compute_given_flowdir(dem, width, height, nodata, flowDir, flowAcc, i, j); // make sure the flowDir is computed
    int dir = flowDir[i * width + j];
    atomic_add(&flowAcc[i * width + j], 1);
    switch(dir){
        case 1:
            acc_recursize(flowDir, flowAcc, i-1, j-1, width, height, nodata, dem);
            break;
        case 2:
            acc_recursize(flowDir, flowAcc, i-1, j, width, height, nodata, dem);
            break;
        case 3:
            acc_recursize(flowDir, flowAcc, i-1, j+1, width, height, nodata, dem);
            break;
        case 4:
            acc_recursize(flowDir, flowAcc, i, j+1, width, height, nodata, dem);
            break;
        case 5:
            acc_recursize(flowDir, flowAcc, i+1, j+1, width, height, nodata, dem);
            break;
        case 6:
            acc_recursize(flowDir, flowAcc, i+1, j, width, height, nodata, dem);
            break;
        case 7:
            acc_recursize(flowDir, flowAcc, i+1, j-1, width, height, nodata, dem);
            break;
        case 8:
            acc_recursize(flowDir, flowAcc, i, j-1, width, height, nodata, dem);
            break;
        default:
            break;
    }
    return;
}



__kernel
void mnt(__global float* dem, const int width, const int height, const int gauche, const int droit, const int nodata, const int cell, volatile __global int* flowDir, volatile __global int* flowAcc){
    int i = get_global_id(0);
    int j = i % width;
    i = i / width;
    compute_given_flowdir(dem, width, height, nodata, flowDir, flowAcc, i, j);
    if (flowDir[i * width + j] != nodata) {
        acc_recursize(flowDir, flowAcc, i, j, width, height, nodata, dem);
    }
}