#include <sys/time.h>
#include<stdio.h>
#include "Helper/TimeProfiler.h"

#define NUM_TIMERS 100

static double get_time(){
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}

static double init = get_time();
static double start_time[NUM_TIMERS];
static double total_time[NUM_TIMERS];
static int counter[NUM_TIMERS];

void timer_start(int i){
	start_time[i] = get_time() - init;
}

void timer_stop(int i){
	double cur = get_time() - init;
	total_time[i] += cur - start_time[i];
	counter[i] += 1;
	if(counter[i]%1000 == 0) fprintf(stderr, "timer %d : %lf / %lf (%lf%%)\n", i, total_time[i], cur, total_time[i] / cur * 100);
}
