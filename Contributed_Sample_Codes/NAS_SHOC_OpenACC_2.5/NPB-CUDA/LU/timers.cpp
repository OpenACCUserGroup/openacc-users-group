#include <stdio.h>
#include <sys/time.h>
#include "main.h"

char *Timers::t_names[t_last];

void Timers::init_timer() {
	t_names[t_total] = "total";
	t_names[t_rhsx] = "rhsx";
	t_names[t_rhsy] = "rhsy";
	t_names[t_rhsz] = "rhsz";
	t_names[t_rhs] = "rhs";
	t_names[t_jacld] = "jacld";
	t_names[t_blts] = "blts";
	t_names[t_jacu] = "jacu";
	t_names[t_buts] = "buts";
	t_names[t_add] = "add";
	t_names[t_l2norm] = "l2norm";
}

Timers::Timers() {
	elapsed = new double [t_last];
	start = new double [t_last];
}

Timers::~Timers() {
	delete[] elapsed;
	delete[] start;
}

void Timers::timer_clear(int n) {
	elapsed[n] = 0.0;
}

void Timers::timer_clear_all() {
	for (int i = 0; i < t_last; i++) elapsed[i] = 0.0;
}

void Timers::timer_start(int n) {
	start[n] = elapsed_time();
}

void Timers::timer_stop (int n) {
	elapsed[n] += elapsed_time() - start[n];
}

double Timers::timer_read(int n) {
	return elapsed[n];
}

double Timers::elapsed_time() {
	//	a generic timer
	static int sec = -1;
	struct timeval tv;
	gettimeofday(&tv, 0L);
	if (sec < 0) sec = tv.tv_sec;
	return (tv.tv_sec - sec) + 1.0e-6*tv.tv_usec;
}

void Timers::timer_print() {
	double trecs[t_last], tmax;
	for (int i = 0; i < t_last; i++) trecs[i] = timer_read(i);
	tmax = trecs[0] == 0.0 ? 1.0 : trecs[0];

	printf("  SECTION     Time (secs)\n");
	for (int i = 0; i < t_last; i++) {
		printf("  %8s:%9.3f  (%6.2f\%)\n", t_names[i], trecs[i], trecs[i]*100./tmax);
		if (i == t_rhs) {
			double t = trecs[t_rhsx] + trecs[t_rhsy] + trecs[t_rhsz];
			printf("     --> %8s:%9.3f  (%6.2f\%)\n", "sub-rhs", t, t*100.0/tmax);
			t = trecs[i] - t;
			printf("     --> %8s:%9.3f  (%6.2f\%)\n", "rest-rhs", t, t*100.0/tmax);
		}
	}
}
