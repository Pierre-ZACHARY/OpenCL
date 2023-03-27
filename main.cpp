//
// Created by o2183251@campus.univ-orleans.fr on 21/03/23.
//

using namespace std;


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "OpenCLTemplate.hpp"
#include <chrono>

bool debug = false;

int* cpu_flow_direction(float* heightMat, int width, int height, int gauche, int droit, int nodata, int cell){
    auto* flowDir = new int[width*height];
#pragma omp parallel for default(none) shared(heightMat, flowDir, width, height, gauche, droit, nodata, cell)
    for (int i=0; i<width; i++) {
        for (int j = 0; j < height; j++) {
            if (heightMat[i * width + j] == nodata) {
                flowDir[i * width + j] = nodata;
            } else {
                // compute the min between self and the 8 neighbors
                float min = heightMat[i * width + j];
                int dir = 0;
                for (int k = -1; k < 2; k++) {
                    for (int l = -1; l < 2; l++) {

                        if (i + k >= 0 && i + k < width && j + l >= 0 && j + l < height) {
                            if(heightMat[(i + k) * width + j + l] == nodata){
                                continue;
                            }
                            if (heightMat[(i + k) * width + j + l] < min) {
                                min = heightMat[(i + k) * width + j + l];
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
    }
    return flowDir;
}

int* cpu_flow_acc(int* flowdir, int width, int height, int nodata){
    int* flowAcc = new int[width*height];
    for(int i=0; i<width*height; i++){
        flowAcc[i] = 0;
    }
    bool haschange = true;
    while(haschange) {
        haschange = false;
        // print that iteration
#pragma omp parallel for default(none) shared(flowAcc, flowdir, width, height, nodata, haschange)
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if(flowAcc[i * width + j] != 0){
                    continue;
                }
                else if (flowdir[i * width + j] == nodata) {
                    flowAcc[i * width + j] = nodata;
                } else {
                    // foreach neighbor, test if any of them is a source of current, if so, add it to the current if the source is not -1 else, do nothing and set haschange to true
                    int sum = 0;
                    bool hasSource = false;
                    bool allSourceComplete = true;
                    for (int k = -1; k < 2; k++) {
                        for (int l = -1; l < 2; l++) {
                            if(k == 0 && l == 0){
                                continue;
                            }
                            if (i + k >= 0 && i + k < width && j + l >= 0 && j + l < height) {
                                if(flowdir[(i + k) * width + j + l]==nodata){
                                    continue;
                                }
                                // test if that neighbor is a source of current
                                bool isSource = false;
                                if(k == -1 && l == -1 && flowdir[(i + k) * width + j + l] == 5){
                                    isSource = true;
                                } else if(k == -1 && l == 0 && flowdir[(i + k) * width + j + l] == 6){
                                    isSource = true;
                                } else if(k == -1 && l == 1 && flowdir[(i + k) * width + j + l] == 7){
                                    isSource = true;
                                } else if(k == 0 && l == 1 && flowdir[(i + k) * width + j + l] == 8){
                                    isSource = true;
                                } else if(k == 1 && l == 1 && flowdir[(i + k) * width + j + l] == 1){
                                    isSource = true;
                                } else if(k == 1 && l == 0 && flowdir[(i + k) * width + j + l] == 2){
                                    isSource = true;
                                } else if(k == 1 && l == -1 && flowdir[(i + k) * width + j + l] == 3){
                                    isSource = true;
                                } else if(k == 0 && l == -1 && flowdir[(i + k) * width + j + l] == 4){
                                    isSource = true;
                                }
                                if (isSource && flowAcc[(i + k) * width + j + l] != 0) {
                                    sum += flowAcc[(i + k) * width + j + l];
                                    hasSource = true;
                                }
                                else if (isSource && flowAcc[(i + k) * width + j + l] == 0){
                                    allSourceComplete = false;
                                    haschange = true;
                                    break;
                                }
                            }
                        }
                        if(!allSourceComplete){
                            break;
                        }
                    }

                    if(allSourceComplete){
                        if (hasSource) {
                            flowAcc[i * width + j] = sum + 1;
                        } else {
                            flowAcc[i * width + j] = 1;
                        }
                    }
                }

            }
        }
    }
    return flowAcc;
}

void printFlowDirFormat(int val){
#ifdef __WINDOWS__
    UINT originalCodePage = GetConsoleOutputCP();
    SetConsoleOutputCP(CP_UTF8);
#endif

    switch(val){
        case 1:
            cout << u8"\u2196" << " ";
            break;
        case 2:
            cout << u8"\u2191" << " ";
            break;
        case 3:
            cout << u8"\u2197" << " ";
            break;
        case 4:
            cout << u8"\u2192" << " ";
            break;
        case 5:
            cout << u8"\u2198" << " ";
            break;
        case 6:
            cout << u8"\u2193" << " ";
            break;
        case 7:
            cout << u8"\u2196" << " ";
            break;
        case 8:
            cout << u8"\u2193" << " ";
            break;
        default:
            cout << val << " ";
            break;
    }

#ifdef __WINDOWS__
    SetConsoleOutputCP(originalCodePage);
#endif
}

void printFlowDirMat(int* flowDirMat, int width, int height){

    for (int i=0; i<height; i++) {
        for (int j = 0; j < width; j++) {
            printFlowDirFormat(flowDirMat[i * width + j]);
        }
        cout << endl;
    }
}


int* cpu(float* heightMat, int width, int height, int gauche, int droit, int nodata, int cell){
    // cpu flow direction algorithm
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    auto* flowDir = cpu_flow_direction(heightMat, width, height, gauche, droit, nodata, cell);

    auto *flowAcc = cpu_flow_acc(flowDir, width, height, nodata);
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "CPU time : " << chrono::duration_cast<chrono::nanoseconds>(end - begin).count() << " ns" << endl;
    if(debug) printFlowDirMat(flowDir, 6, 6);
    return flowAcc;
}

int* gpu(float* heightMat, int width, int height, int gauche, int droit, int nodata, int cell, int* flowDir){
    // gpu flow direction algorithm
    auto* arr = new int[width*height];
    for(int i=0; i<width*height; i++){
        arr[i] = -1;
    }
    auto* openCLTemplate = new OpenCLTemplate();
    openCLTemplate->setSourceFile("mnt.cl", "mnt")
            ->addBuffer<float>(width*height)->write(heightMat)
            ->addConst<int>(width)
            ->addConst<int>(height)
            ->addConst<int>(gauche)
            ->addConst<int>(droit)
            ->addConst<int>(nodata)
            ->addConst<int>(cell)
            ->addBuffer<int>(width*height)->write(arr) // flowdir
            ->addBuffer<int>(width*height)->write(new int[width*height]{0}); // flowacc

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
//    openCLTemplate->runAuto(width*height);
    // solution pour le cas où la carte manque de mémoire : on divise l'entrée en chunks de 128*128 ( par exemple ) et on les traite séparément
    int chunk_size = min(1024*1024, width*height);
    int num_step = ceil(((double) width*height)/((double) 1024*1024));
    for(int i=0; i<num_step; i++){
        int numworkers = min(chunk_size, (width*height)-i*chunk_size);
        openCLTemplate->runAuto(numworkers, cl::NDRange(i*chunk_size));
    }
    int* flowDirRes = openCLTemplate->getBuffer<int>(1)->read();
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "GPU time : " << chrono::duration_cast<chrono::nanoseconds>(end - begin).count() << " ns" << endl;

    memcpy(flowDir, flowDirRes, width*height*sizeof(int));
    if(debug) printFlowDirMat(flowDir, 6, 6);
    auto* flowAcc = openCLTemplate->getBuffer<int>(2)->read();
    return flowAcc;
}


int main(){
    string filename = "grd_618360_6754408_2.txt";
    FILE *fp = fopen(filename.c_str(),"r");
    if (fp==NULL){
        printf("Error opening file %s",filename.c_str());
        exit(0);
    }

    int nx, ny, gauche, droit, cell, nodata;

    fscanf(fp,"%d",&nx);
    fscanf(fp,"%d",&ny);
    fscanf(fp,"%d",&gauche);
    fscanf(fp,"%d",&droit);
    fscanf(fp,"%d",&cell);
    fscanf(fp,"%d",&nodata);

    float hauteur;
    auto* tab = new float[nx*ny];
    for (int i=0; i<nx; i++) {
        for (int j=0; j<ny; j++) {
            fscanf(fp,"%f",&hauteur);
            tab[i*nx+j] = hauteur;
        }
    }
    fclose(fp);
    auto* flowDir = cpu_flow_direction(tab, nx, ny, gauche, droit, nodata, cell);

    int* res = cpu(tab, nx, ny, gauche, droit, nodata, cell);
    // print res
    if(debug){
        for (int i=0; i<6; i++) {
            for (int j=0; j<6; j++) {
                cout << res[i*nx+j] << " ";
            }
            cout << endl;
        }
    }

    int* flowDirGPU = new int[nx*ny];
    int* res2 = gpu(tab, nx, ny, gauche, droit, nodata, cell, flowDirGPU);
    if(debug){
        for (int i=0; i<6; i++) {
            for (int j=0; j<6; j++) {
                cout << res2[i*nx+j] << " ";

            }
            cout << endl;
        }
    }


    // check if flowdir and floatacc are the same on the cpu and gpu solution
    for(int i=0; i<nx; i++){
        for(int j=0; j<ny; j++) {
            if (flowDir[i * nx + j] != flowDirGPU[i * nx + j]) {
                cout << "KO at i:" << i << ", j:" << j << " cpu = " << flowDir[i * nx + j] << " gpu = "
                     << flowDirGPU[i * nx + j] << endl;

                // debug ( check if flowdir is correct at this point on the gpu / cpu solution )
//                cout << "Flowdir CPU:" << endl;
//
//                for (int k=i-5; k<i+5; k++) {
//                    for (int l=j-5; l<j+5; l++) {
//                        if(k>=0 && k<nx && l>=0 && l<ny){
//                            if(k==i && l==j) cout << "X";
//                            printFlowDirFormat(flowDir[k*nx+l]);
//                        }
//                    }
//                    cout << endl;
//                }
//
//                cout << "Flowdir GPU:" << endl;
//
//                for (int k=i-5; k<i+5; k++) {
//                    for (int l=j-5; l<j+5; l++) {
//                        if(k>=0 && k<nx && l>=0 && l<ny){
//                            if(k==i && l==j) cout << "X";
//                            printFlowDirFormat(flowDirGPU[k*nx+l]);
//                        }
//                    }
//                    cout << endl;
//                }
                return 1;
            }
        }
    }


    return 0;
}