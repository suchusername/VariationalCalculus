// Serial Whale Optimization Algorithm for Calculus of Variations Problems (SWOACVP) source code 
// 2018-12-08
// H.H. Mehne, S. Mirjalili, A Direct Method for Solving Calculus of Variations Problems using the Whale Optimization Algorithm, Example 2. 
// Compile the file in linux command line with "g++ -o calvar2 calvar2.cpp" 
// Run it in linux command with  "./calvar2"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iomanip>
#include <omp.h>

#define SearchAgents_no 500
#define Max_iter 500
#define lb1 -2.0
#define ub1 2.0
#define lb2 -2.0
#define ub2 2.0
#define dim 100
#define pi 3.14159265359
#define T_f 9*pi/8
#define x1f sin(9*pi/8)
#define x2f -sin(9*pi/8)

        
typedef struct {
float positions1[dim+1];
float positions2[dim+1];
float xdot1[dim+1];	
float xdot2[dim+1];	
float fitness;
} whale;
whale sample[SearchAgents_no];

float Leader_pos1[dim+1], Leader_pos2[dim+1];
float Leader_score=10000;
int Leader_whale;
float a, a2; 
float delta_t=(float) T_f/dim;
float performance_index[Max_iter];

float func(float x1, float x2, float x1dot, float x2dot) // (x1)' = func1(x1,x2,u)
{ 
return 2*x1*x2 + x1dot*x1dot + x2dot*x2dot;
}

void MakeInput4TecPlot(char* filename1, char* filename2, float* u1, float* u2, float* perf) // This Subroutine makes input file for drawing the results in Tecplot Software
{
	int i;
	FILE *OutPut;
	float t, DTM1, DTM2, sum1, sum2, dotx1, dotx2; 
	
	OutPut=fopen(filename1,"w"); //Writing results to the first file 
	fprintf(OutPut, "title = \"SWOACVP  Output Example2\" \n");
	fprintf(OutPut,"variables= \"t\", \"x11\", \"x22\", \"x1Exact\", \"x2Exact\" \n");
	fprintf(OutPut, "zone\n");
	sum1=0.0;
	sum2=0.0;
	for(i=0;i<=dim;i++){
		t = i*delta_t;
		DTM1 = 1.0003*(t)-0.1667*pow(t,3) + 0.0083*pow(t,5) -0.0002*pow(t,7);
		DTM2 = -1.0003*(t)+0.1667*pow(t,3) - 0.0083*pow(t,5) +0.0002*pow(t,7);	
		if (i<dim){
		dotx1=(u1[i+1]-u1[i])/delta_t;
		dotx2=(u2[i+1]-u2[i])/delta_t;
		sum1+=delta_t*func(sin(t),-sin(t),-cos(t),cos(t));
		sum2+=delta_t*func(u1[i],u2[i],dotx1,dotx2);
		}
		fprintf(OutPut, "%4f %4f %4f %4f %4f\n", t, u1[i], u2[i], sin(t), -sin(t));
	}
	printf("\n The optimum value in the case of Euler-Lagrange condition =%f, \n The optimum value in the non-diferentiable case =%f\n", sum1, sum2);
	fclose(OutPut);
	
	OutPut=fopen(filename2,"w"); //Writing obtained performance indices to the second file 
	fprintf(OutPut, "title = \"SWOACVP Performance Index Example2\" \n");
	fprintf(OutPut," \"u\", \"Performance\"\n");
	fprintf(OutPut, "zone    i=%d\n", Max_iter);
	for(i=0;i<Max_iter;i++)
		fprintf(OutPut, "%d %4f\n", i, perf[i]);
	fclose(OutPut);
	
}

void initialization() // This subroutine provides an initial distribution for each whale 
{
int i,j;
	for(i=0;i<SearchAgents_no;i++){
		for(j=1;j<dim;j++){
	      sample[i].positions1[j]=lb1+(rand()/(RAND_MAX+1.0))*(ub1-lb1);
	      sample[i].positions2[j]=lb2+(rand()/(RAND_MAX+1.0))*(ub2-lb2);
		}
sample[i].positions1[0]=0.0; // initial condition x1(0)
sample[i].positions1[dim]=x1f; // initial condition x1(T_f)
sample[i].positions2[0]=0.0; // initial condition x2(0)
sample[i].positions2[dim]=x2f; // initial condition x2(T_f)
}
}

void regulation() // This subroutine checks the fitting of the solution and fit the out of bound values to [l_b, u_b] 
{
int i,j;
int Flag4ub, Flag4lb;

	for(i=0;i<SearchAgents_no;i++){
		for(j=0;j<dim;j++){
		Flag4ub=( sample[i].positions1[j] > ub1 ? 1 : 0 );
		Flag4lb=( sample[i].positions1[j] < lb1 ? 1 : 0 );
		sample[i].positions1[j]=sample[i].positions1[j]*(!(Flag4ub+Flag4lb))+ub1*Flag4ub+lb1*Flag4lb;
			}
}
	for(i=0;i<SearchAgents_no;i++){
		for(j=0;j<dim;j++){
		Flag4ub=( sample[i].positions2[j] > ub2 ? 1 : 0 );
		Flag4lb=( sample[i].positions2[j] < lb2 ? 1 : 0 );
		sample[i].positions2[j]=sample[i].positions2[j]*(!(Flag4ub+Flag4lb))+ub2*Flag4ub+lb2*Flag4lb;
			}
}

}

void smoothing() // This subroutine shmooths the corners of the solution by averaging 
{
int i,j;

	for(i=0;i<SearchAgents_no;i++){
		for(j=1;j<dim;j++){
		if (( sample[i].positions1[j] - sample[i].positions1[j-1] )*( sample[i].positions1[j+1] - sample[i].positions1[j] )<0);
			sample[i].positions1[j]=(sample[i].positions1[j-1]+sample[i].positions1[j+1])/2;
			}
}
for(i=0;i<SearchAgents_no;i++){
		for(j=1;j<dim;j++){
		if (( sample[i].positions2[j] - sample[i].positions2[j-1] )*( sample[i].positions2[j+1] - sample[i].positions2[j] )<0);
			sample[i].positions2[j]=(sample[i].positions2[j-1]+sample[i].positions2[j+1])/2;
			}
}
}


float IndexEvaluation(int whale_no) // This subroutine calculates the performance index of an indivisual whale 
{
int i,j,m,n;
float sum,k1,k2,k3,k4,h1,h2,h3,h4,x1,x2, x3, x4, u1, u2;
float m1, m2, m3, m4, p1, p2, p3, p4;
sample[whale_no].positions1[0]=0.0; // initial condition x1(0)
sample[whale_no].positions1[dim]=x1f; // initial condition x1(T_f)
sample[whale_no].positions2[0]=0.0; // initial condition x2(0)
sample[whale_no].positions2[dim]=x2f; // initial condition x2(T_f)

sum=0.0;

for(i=0;i<dim;i++) {
	sample[whale_no].xdot1[i] = (sample[whale_no].positions1[i+1] - sample[whale_no].positions1[i])/delta_t;
	sample[whale_no].xdot2[i] = (sample[whale_no].positions2[i+1] - sample[whale_no].positions2[i])/delta_t;
	sum+= delta_t*func(sample[whale_no].positions1[i],sample[whale_no].positions2[i],sample[whale_no].xdot1[i],sample[whale_no].xdot2[i]);
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
    best_whale = Leader_whale;
	for(i=0;i<SearchAgents_no;i++)
		if (sample[i].fitness<Leader_score){ // Change this to > for maximization problem
            	best_whale = i;
				Leader_score=sample[best_whale].fitness;
    	for(j=0;j<=dim;j++) {
			Leader_pos1[j]=sample[best_whale].positions1[j];// copy sample[i].positions1 to Leader_pos1
			Leader_pos2[j]=sample[best_whale].positions2[j];// copy sample[i].positions2 to Leader_pos2
			}
		}
	Leader_whale=best_whale;
	
}

void position_update() // This subroutine updates the solution based on Eq. (15) of the paper 
{
	int i,j, k, rand_leader_index1, rand_leader_index2; 
	float r1, r2, A, C, b, p, l, D_X_rand1, D_Leader1, distance2Leader1;
	float D_X_rand2, D_Leader2, distance2Leader2;
	float X_rand1[dim], X_rand2[dim];
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
                    rand_leader_index1 = (int) floor(SearchAgents_no*(rand()/(RAND_MAX+1.0)));
					rand_leader_index2 = (int) floor(SearchAgents_no*(rand()/(RAND_MAX+1.0)));
					for (k=0;k<dim;k++)  {
						X_rand1[k] = sample[rand_leader_index1].positions1[k];
						X_rand2[k] = sample[rand_leader_index2].positions2[k];
					}
	for (j=1;j<dim;j++){				
					D_X_rand1=fabs(C*X_rand1[j]-sample[i].positions1[j]); 
					D_X_rand2=fabs(C*X_rand2[j]-sample[i].positions2[j]); 
                    sample[i].positions1[j]=X_rand1[j]-A*D_X_rand1;      // Eq.(15) the lower relation
					sample[i].positions2[j]=X_rand2[j]-A*D_X_rand2;      // Eq.(15) the lower relation
				}
                }
               else if (abs(A)<1) {
				   for (j=1;j<dim;j++){				
                    D_Leader1=fabs(C*Leader_pos1[j]-sample[i].positions1[j]); 
					D_Leader2=fabs(C*Leader_pos2[j]-sample[i].positions2[j]); 
                    sample[i].positions1[j]=Leader_pos1[j]-A*D_Leader1;      // Eq. (15) the upper relation
					sample[i].positions2[j]=Leader_pos2[j]-A*D_Leader2;      // Eq. (15) the upper relation
                }
			   }
                
            else if (p>=0.5){
				if (abs(A)<1) {
              for (j=1;j<dim;j++){	
                distance2Leader1=fabs(Leader_pos1[j]-sample[i].positions1[j]);       
				distance2Leader2=fabs(Leader_pos2[j]-sample[i].positions2[j]);       
                sample[i].positions1[j]=distance2Leader1*exp(b*l)*cos(l*2*pi)+Leader_pos1[j]; // Eq. (15) the middle relation
				sample[i].positions2[j]=distance2Leader2*exp(b*l)*cos(l*2*pi)+Leader_pos2[j]; // Eq. (15) the middle relation
			}
				}
				            }
sample[i].positions1[0]=0.0; // initial condition x1(0)
sample[i].positions1[dim]=x1f; // initial condition x1(T_f)
sample[i].positions2[0]=0.0; // initial condition x2(0)
sample[i].positions2[dim]=x2f; // initial condition x2(T_f)
}
}

int main( int argc, char *argv[])
{

time_t seconds;
int t;
double wtime;
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
	performance_index[t]=Leader_score;
	printf("\n t=%d, Leader=%f, Whale=%d",t, Leader_score,Leader_whale);
		t++;
	
} while (t<Max_iter);
MakeInput4TecPlot((char*) "outputcalcvarEx2.dat", (char*) "PerformanceEx2.dat", Leader_pos1, Leader_pos2, performance_index);		
return 0;
}
