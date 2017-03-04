#include <stdio.h>
#include "CTimer.h"
#include <iostream>


void
DoReduceFloatsIters( unsigned int nIters,
                        void* ivdata, 
                        unsigned int nItems, 
                        void* ovres, 
                        double* itersReduceTime,
                        double* totalReduceTime,
                        void (*reducefunc)( void* localsum, void* result ) )
{
    std::cout << "DoReduceFloatsIters" << std::endl;    
    float sum = 0.0;
    float* __restrict__ idata = (float*)ivdata;
    float* ores = (float*)ovres;

    // start a timer that includes the transfer time and iterations
    int wholeTimerHandle = Timer_Start();

    #pragma acc data copyin(idata[0:nItems])
    {

        // now that we've copied data to device,
        // time the iterations.
        int iterTimerHandle = Timer_Start();

        // note: do *not* try to map the iterations loop to the accelerator
        for( unsigned int iter = 0; iter < nIters; iter++ )
        {
            sum = 0.0;

            #pragma omp parallel for reduction(+:sum) default(none) shared(nItems,idata)
            //#pragma acc loop reduction( +:sum ) independent 
            #pragma acc kernels loop reduction( +:sum ) 
            for( unsigned int i = 0; i < nItems; i++ )
            {
                sum += idata[i];
            }

            // we may have to reduce further
            if( reducefunc != 0 )
            {
                float res;
                (*reducefunc)( &sum, &res );
                sum = res;
            }
        }

        // stop the timer and record the result (in seconds)
        *itersReduceTime = Timer_Stop( iterTimerHandle, "" );

    } /* end acc data region for idata */

    *totalReduceTime = Timer_Stop( wholeTimerHandle, "" );

    // save the result
    *ores = sum;
}




void
DoReduceDoublesIters( unsigned int nIters,
                        void* ivdata, 
                        unsigned int nItems, 
                        void* ovres, 
                        double* itersReduceTime,
                        double* totalReduceTime,
                        void (*reducefunc)( void* localsum, void* result ) )
{
    std::cout << "DoReduceDoublesIters" << std::endl;
    double sum = 0.0;
    double* __restrict__ idata = (double*)ivdata;
    double* ores = (double*)ovres;

    // start a timer that includes both transfer time and iterations
    int wholeTimerHandle = Timer_Start();

    #pragma acc data copyin(idata[0:nItems])
    {

        // now that we've copied data to device,
        // time the iterations.
        int iterTimerHandle = Timer_Start();

        // note: do *not* try to map the iterations loop to the accelerator
        for( unsigned int iter = 0; iter < nIters; iter++ )
        {
            sum = 0.0;

            #pragma omp parallel for reduction(+:sum) default(none) shared(nItems, idata)
            //#pragma acc loop reduction( +:sum ) independent
            #pragma acc kernels loop reduction( +:sum ) 
            for( unsigned int i = 0; i < nItems; i++ )
            {
                sum += idata[i];
            }

            // we may have to reduce further
            if( reducefunc != 0 )
            {
                double res;
                (*reducefunc)( &sum, &res );
                sum = res;
            }
        }

        // stop the timer and record the result (in seconds)
        *itersReduceTime = Timer_Stop( iterTimerHandle, "" );

    } /* end acc data region for idata */

    *totalReduceTime = Timer_Stop( wholeTimerHandle, "" );

    // save the result
    *ores = sum;
}



