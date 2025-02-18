#ifndef TIMER_H
#define TIMER_H

#include <stdlib.h>
#include <sys/time.h>

struct timeval timerStart;

void StartTimer()
{
    gettimeofday(&timerStart, NULL);
}

double GetTimer()
{
    struct timeval timerEnd;
    gettimeofday(&timerEnd, NULL);
    
    double elapsed = (timerEnd.tv_sec - timerStart.tv_sec) * 1000.0;   // Convert seconds to milliseconds
    elapsed += (timerEnd.tv_usec - timerStart.tv_usec) / 1000.0;       // Convert microseconds to milliseconds
    
    return elapsed;
}

#endif /* TIMER_H */
