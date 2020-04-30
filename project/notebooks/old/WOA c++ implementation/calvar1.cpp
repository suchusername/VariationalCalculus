// Serial Whale Optimization Algorithm for Calculus of Variations Problems (SWOACVP) source code 
// 2018-12-08
// H.H. Mehne, S. Mirjalili, A Direct Method for Solving Calculus of Variations Problems using the Whale Optimization Algorithm, Example 1. 
// Compile the file in linux command line with "g++ -o calvar1 calvar1.cpp" 
// Run it in linux command with  "./calvar1"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <iomanip>
#include <omp.h>

#define SearchAgents_no 400
#define Max_iter 500
#define lb -2.0
#define ub 0.5
#define dim 50
#define pi 3.1415
#define T_f 1.0
        
typedef struct {
float positions[dim+1];
float xdot[dim+1];	
float fitness;
} whale;
whale sample[SearchAgents_no];

float Leader_pos[dim];
float Leader_score=10000;
int Leader_whale = 0;
float a, a2; 
float delta_t=(float) T_f/dim;
float performance_index[Max_iter];


void MakeInput4TecPlot(char* filename1, char* filename2,float* u, float* perf) // This Subroutine makes input file for drawing the results in Tecplot Software
{
	int i;
	float t;
	FILE *OutPut;
	
	OutPut=fopen(filename1,"w"); //Writing the results to the first file 
	fprintf(OutPut, "title = \"SWOACVP Output Example1\" \n");
	fprintf(OutPut," variables= \"t\", \"x\", \"DTM\", \"exact\" \n");
	fprintf(OutPut, "zone\n");
	for(i=0;i<=dim;i++){
		t= i*delta_t;
				fprintf(OutPut, "%4f %4f %4f %4f\n", t, u[i], 0.443778*t - pow(t,2)/4 + 0.0739631*pow(t,3) - pow(t,4)/48 +  0.00369815*pow(t,5) - pow(t,6)/1440 + 0.0000880513*pow(t,7), 0.5+(2-exp(1))*exp(t)/(4*(exp(2)-1))+(exp(1)-2*exp(2))*exp(-t)/(4*(exp(2)-1)));
	}
	fclose(OutPut);
	
	OutPut=fopen(filename2,"w"); //Writing obtained performance indices to the second file 
	fprintf(OutPut, "title = \"SWOACVP Performance Index Example1\" \n");
	fprintf(OutPut," \"u\", \"Performance\"\n");
	fprintf(OutPut, "zone\n");
	for(i=0;i<Max_iter;i++)
		fprintf(OutPut, "%d %4f\n", i, perf[i]);
	fclose(OutPut);
	
}
void initialization() // This subroutine provides an initial distribution for each whale 
{
int i,j;
	for(i=0;i<SearchAgents_no;i++){
		for(j=1;j<dim;j++)
	      sample[i].positions[j]=lb+(rand()/(RAND_MAX+1.0))*(ub-lb);
	      sample[i].positions[0]=0.0;
		  sample[i].positions[dim]=0.25;
	}
		
}

void regulation() // This subroutine checks the solution and fit the out of bound values to [l_b, u_b] 
{
int i,j;
int Flag4ub, Flag4lb;

	for(i=0;i<SearchAgents_no;i++){
		for(j=0;j<dim;j++){
		Flag4ub=( sample[i].positions[j] > ub ? 1 : 0 );
		Flag4lb=( sample[i].positions[j] < lb ? 1 : 0 );
		sample[i].positions[j]=sample[i].positions[j]*(!(Flag4ub+Flag4lb))+ub*Flag4ub+lb*Flag4lb;
				}
}
}

void smoothing() // This subroutine shmooths the corners of the solution values by averaging 
{
int i,j;

	for(i=0;i<SearchAgents_no;i++){
		for(j=1;j<dim;j++){
		if (( sample[i].positions[j] - sample[i].positions[j-1] )*( sample[i].positions[j+1] - sample[i].positions[j] )<0);
			sample[i].positions[j]=(sample[i].positions[j-1]+sample[i].positions[j+1])/2;
			}
}
}


float func(float t, float x, float xdot) // (x1)' = func1(x1,x2,u)
{ 
return x*x+t*xdot+xdot*xdot;
}


float IndexEvaluation(int whale_no) // This subroutine calculates the performance index of an indivisual whale 
{
int i,j;
float sum,k1,k2,k3,k4,h1,h2,h3,h4,x1,x2,u;
sample[whale_no].positions[0]=0.0; // initial condition x1(0)=0.0
sample[whale_no].positions[dim]=0.25; // initial condition x2(0)=0.25
sum=0.0;
for(i=0;i<dim;i++) {
	sample[whale_no].xdot[i] = (sample[whale_no].positions[i+1] - sample[whale_no].positions[i])/(delta_t);
    sum+= delta_t*func(delta_t*i,sample[whale_no].positions[i],sample[whale_no].xdot[i]);
				}
return sum;
}

void fitness_evaluation() // This subroutine evaluates the performance index of all search agents(whales) 
{
int i;
	for(i=0;i<SearchAgents_no;i++)
			sample[i].fitness=IndexEvaluation(i);
}

void finding_local_opt() // This subroutine finds the whale with the best performance index 
{
int i,j, best_whale;
best_whale=Leader_whale;
	for(i=0;i<SearchAgents_no;i++)
		if (sample[i].fitness<Leader_score){ // Change this to > for maximization problem
            Leader_score=sample[i].fitness; 
			best_whale = i;
			for(j=0;j<=dim;j++) Leader_pos[j]=sample[i].positions[j];// Copy sample[i].positions to Leader_pos
			}
			Leader_whale=best_whale;
}
void position_update() // This subroutine updates the solution on Eq. (15) of the paper 
{
	int i,j, k, rand_leader_index; 
	float r1, r2, A, C, b, p, l, D_X_rand, D_Leader, distance2Leader;
	float X_rand[dim];
	for (i=1;i<SearchAgents_no;i++){
        r1=rand()/(RAND_MAX+1.0); // r1 is a random number in [0,1]
        r2=rand()/(RAND_MAX+1.0); // r2 is a random number in [0,1]
        
        A=2*a*r1-a;  
        C=2*r2;      
        
        
        b=1;               
        l=(a2-1)*rand()/(RAND_MAX+1.0)+1;   
        
        p = rand()/(RAND_MAX+1.0);        
        
                 
            if (p<0.5)
                if (fabs(A)>=1){
                    rand_leader_index = (int) floor(SearchAgents_no*(rand()/(RAND_MAX+1.0)));
					for (k=0;k<dim;k++)  X_rand[k] = sample[rand_leader_index].positions[k];
					for (j=1;j<dim;j++){
					D_X_rand=fabs(C*X_rand[j]-sample[i].positions[j]); 
                    sample[i].positions[j]=X_rand[j]-A*D_X_rand;      // Eq.(15) the lower relation
				}
                }
                else if (fabs(A)<1) {
					for (j=1;j<dim;j++){
                    D_Leader=fabs(C*Leader_pos[j]-sample[i].positions[j]); 
                    sample[i].positions[j]=Leader_pos[j]-A*D_Leader;      // Eq. (9) the upper relation
                }
				}
                
            else if (p>=0.5){
              for (j=1;j<dim;j++){
                distance2Leader=fabs(Leader_pos[j]-sample[i].positions[j]);       
                sample[i].positions[j]=distance2Leader*exp(b*l)*cos(l*2*pi)+Leader_pos[j];  // Eq. (15) the middel relation
			}
            }
 
sample[i].positions[0]=0.0;
sample[i].positions[dim]=0.25;
; 
}
//Leader_pos[dim]=0.25;
}

int main( int argc, char *argv[])
{

time_t seconds;
int t;
srand(uint(seconds));
initialization();
t=0;
do {
    regulation();
	smoothing();
	fitness_evaluation();
	finding_local_opt();
	a=2-t*((2)/Max_iter);     // a decreases linearly fron 2 to 0 
    a2=-1+t*((-1)/Max_iter); // a2 linearly dicreases from -1 to -2 
	position_update();
	printf("\n t=%d, Leader=%f, Whale=%d",t, Leader_score,Leader_whale);
	performance_index[t]=Leader_score;
	t++;
	
} while (t<Max_iter);

MakeInput4TecPlot((char*) "outputsEx1.dat", (char*) "PerformanceEx1.dat", Leader_pos, performance_index);

return 0;
}
