#include <stdio.h>
#include <math.h>


#define DT_scan 1.0
#define dT_scan 0.001

#define DT 0.05
#define tol_T 0.0000001
#define tol_Z1 0.0001

#define MAX_D  5
#define MAX_E  100000

double  energy[MAX_D][MAX_E], hist_E[MAX_D][MAX_E], hist_M[MAX_D][MAX_E], hist_M2[MAX_D][MAX_E], hist_M4[MAX_D][MAX_E]; /* histogram method */
int     SIZE_E[MAX_D], SIZE_D;

double  T0;
double  T1[MAX_D];                /* Original Temperature */
long    Nsite;
double  Z1[MAX_D]; 

void read_hist()
{                              
   FILE *histfile; char dum[50],dum2[20];
   double tmp1, tmp2, tmp3, tmp4, tmp5;
   int i, D, D0;

   T0 = 0.; D=0;
   for (D0=0;D0<10;D0++)
   {
      sprintf(dum2,"hist%1d.in",D0);
      histfile=fopen(dum2,"r");
      if ( histfile == NULL ) continue;

      if (D >= MAX_D )
      {
         printf("Too small MAX_D = %d",MAX_D);
         exit(1);
      }

      printf("#Reading file %s\n",dum2);

      fscanf(histfile,"%s  %lf  %s  %ld\n",dum,&tmp1,dum2,&Nsite);
      T1[D] = tmp1;
      T0 += T1[D];
      fgets(dum,80,histfile);

      printf("#T1 = %lf  Nsite = %ld\n",T1[D],Nsite); 

      for (i=0;i<MAX_E;i++)
      {
         if (fscanf(histfile,"%lf %lf %lf %lf %lf\n",&tmp1,&tmp2,&tmp3,&tmp4,&tmp5) == EOF) break;
         energy[D][i] = tmp1; hist_E[D][i] = tmp2;
         hist_M[D][i] = tmp3; hist_M2[D][i] = tmp4; hist_M4[D][i] = tmp5;
      }
      SIZE_E[D]=i;
      if (SIZE_E[D]==MAX_E)
      {
         printf("Too small MAX_E = %d",MAX_E);
         exit(1);
      }
      fclose(histfile);
      D++;
   } /* for:D */

   SIZE_D=D;
   if ( SIZE_D == 0 )
   {
       printf("Error reading file hist?.in\n");
       exit(1);
   }
   T0 = T0 / SIZE_D;
}

void get_Z1()
{
   int i, j, k, D, iter;
   double Z2[MAX_D], s_N, s_Z, tot_Z, diff_Z;

   for (D=0;D<SIZE_D;D++)
   {
     Z1[D] = 1.0;
   }

   if (SIZE_D <= 1) return;

   for (iter=1;iter<=50;iter++)
   {
      tot_Z = 0.;
      for (D=0;D<SIZE_D;D++)
      {
         Z2[D] = 0.;
         for (i=0;i<SIZE_D;i++)
         for (k=0;k<SIZE_E[i];k++)
         {
            for (s_Z = 0., j=0;j<SIZE_D;j++)
            {
               s_Z += exp((1./T1[D] - 1./T1[j])*energy[i][k])/Z1[j];
            }
            Z2[D] += hist_E[i][k] / s_Z;
         }
         tot_Z += Z2[D];
      }
      diff_Z = 0.;
      for (D=0;D<SIZE_D;D++)
      {
        Z2[D] = Z2[D]/tot_Z;
        diff_Z += abs(Z2[D] - Z1[D]);
        Z1[D] = Z2[D];
      }
      if (diff_Z < tol_Z1) break;
   }
   printf("After %d iteration, Z_i is obtained.\n",iter);
   for (D=0;D<SIZE_D;D++) printf("%le ",Z1[D]); printf("\n");
}

void get_avg(double T, double *s_e, double *s_e2, double *s_m, double *s_m2, double *s_m4, double *s_me, double *s_m2e, double *s_m4e)
{
   int i, D;
   double e_1, m_1, m_2, m_4, Z, Z_1;

   *s_e = *s_e2 = *s_m = *s_m2 = *s_m4 = *s_me = *s_m2e = *s_m4e = Z = 0.;
   for (D=0;D<SIZE_D;D++)
   for (i=0;i<SIZE_E[D];i++)
   {
      e_1   = energy[D][i]/(double)Nsite;
      m_1   = hist_M[D][i];
      m_2   = hist_M2[D][i];
      m_4   = hist_M4[D][i];
      Z_1   = Z1[D] * hist_E[D][i]*exp(-energy[D][i]*(1./T-1./T1[D])) ;

      *s_e  += e_1 * Z_1 ;
      *s_e2 += (e_1*e_1) * Z_1 ;
      *s_m  += m_1 * Z_1 ;
      *s_m2 += m_2 * Z_1 ;
      *s_m4 += m_4 * Z_1 ;
      *s_me += (m_1*e_1) * Z_1 ;
      *s_m2e += (m_2*e_1) * Z_1 ;
      *s_m4e += (m_4*e_1) * Z_1 ;

      Z    += Z_1 ;
   }
   *s_e  = *s_e/Z;
   *s_e2 = *s_e2/Z;
   *s_m  = *s_m/Z;
   *s_m2 = *s_m2/Z;
   *s_m4 = *s_m4/Z;
   *s_me  = *s_me/Z;
   *s_m2e = *s_m2e/Z;
   *s_m4e = *s_m4e/Z;
}

double Cv(double T)
{
   double s_e, s_e2, s_e3, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e;

   get_avg(T, &s_e, &s_e2, &s_m, &s_m2, &s_m4, &s_me, &s_m2e, &s_m4e);

   return (double)Nsite*(s_e2-s_e*s_e)/(T*T);
}

double chi(double T)
{
   double s_e, s_e2, s_e3, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e;

   get_avg(T, &s_e, &s_e2, &s_m, &s_m2, &s_m4, &s_me, &s_m2e, &s_m4e);

   return (double)Nsite*(s_m2-s_m*s_m)/(T);
}

double dm_dbeta(double T)
{
   double s_e, s_e2, s_e3, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e;

   get_avg(T, &s_e, &s_e2, &s_m, &s_m2, &s_m4, &s_me, &s_m2e, &s_m4e);

   return (double)Nsite*(s_m*s_e - s_me);
}

double binder(double T)
{
   double s_e, s_e2, s_e3, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e;

   get_avg(T, &s_e, &s_e2, &s_m, &s_m2, &s_m4, &s_me, &s_m2e, &s_m4e);

   return 1.0 - s_m4/(s_m2*s_m2*3.0);
}

double dbinder_dbeta(double T)
{
   double s_e, s_e2, s_e3, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e;

   get_avg(T, &s_e, &s_e2, &s_m, &s_m2, &s_m4, &s_me, &s_m2e, &s_m4e);

   return (double)Nsite*(s_m2*(s_m4e+s_m4*s_e) - 2.0*s_m4*s_m2e)/(3.0*s_m2*s_m2*s_m2);
}

void scan_T()
{
   FILE *out;
   int i;
   double s_e, s_e2, s_e3, s_m, s_m2, s_m4, s_me, s_m2e, s_m4e, T;

   out = fopen("hist.out","w");
   fprintf(out,"#T         m           chi         E         Cv     dm/dbeta binder    dbinder/dbeta\n");

   for (T=T0-DT_scan;T<=T0+DT_scan;T+=dT_scan)
   {
      get_avg(T, &s_e, &s_e2, &s_m, &s_m2, &s_m4, &s_me, &s_m2e, &s_m4e);
      fprintf(out,"%le %le %le %le %le %le %le %le\n",T, s_m, (double)Nsite*(s_m2-s_m*s_m)/(T),
                 s_e, (double)Nsite*(s_e2-s_e*s_e)/(T*T),
                 (double)Nsite*(s_m*s_e - s_me),
                 1.0 - s_m4/(s_m2*s_m2*3.0), (double)Nsite*(s_m2*(s_m4e+s_m4*s_e) - 2.0*s_m4*s_m2e)/(3.0*s_m2*s_m2*s_m2) ) ;
   }
   fclose(out);
}

/***************************************************************************/
/* Finds the maximum value of a function f                                 */
/* Modified from "golden" in Numerical Recipes Ch.10                       */
/* Input : Given a function f(x), ax<bx<cx and f(bx)>f(ax) and f(bx)>f(cx) */
/* Output : Maximum value = golden = f(xmax)                               */
/***************************************************************************/
#define R 0.61803399
#define C (1.0-R)
#define SHFT2(a,b,c) (a)=(b);(b)=(c);
#define SHFT3(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);
double golden(double ax, double bx, double cx, double (*f)(double), double tol,
	double *xmax)
{
	double f1,f2,x0,x1,x2,x3;

	x0=ax;
	x3=cx;
	if (fabs(cx-bx) < fabs(bx-ax)) {
		x1=bx;
		x2=bx+C*(cx-bx);
	} else {
		x2=bx;
		x1=bx-C*(bx-ax);
	}
	f1=(*f)(x1);
	f2=(*f)(x2);
	while (fabs(x3-x0) > tol*(fabs(x1)+fabs(x2))) {
		if (f2 > f1) {
			SHFT3(x0,x1,x2,R*x1+C*x3)
			SHFT2(f1,f2,(*f)(x2))
		} else {
			SHFT3(x3,x2,x1,R*x2+C*x0)
			SHFT2(f2,f1,(*f)(x1))
		}
	}
	if (f1 > f2) {
		*xmax=x1;
		return f1;
	} else {
		*xmax=x2;
		return f2;
	}
}
#undef C
#undef R
#undef SHFT2
#undef SHFT3

int main (int argc,char** argv)
{
   FILE *out;
   int i;
   double Cv_max, T_Cv_max, chi_max, T_chi_max;
   double dm_dbeta_max, T_dm_dbeta_max, dbinder_dbeta_max, T_dbinder_dbeta_max;

   read_hist();
   get_Z1();
   scan_T();

   chi_max           = golden(T0-DT,T0,T0+DT,chi          ,tol_T,&T_chi_max);
   Cv_max            = golden(T0-DT,T0,T0+DT,Cv           ,tol_T,&T_Cv_max);
   dm_dbeta_max      = golden(T0-DT,T0,T0+DT,dm_dbeta     ,tol_T,&T_dm_dbeta_max);
   dbinder_dbeta_max = golden(T0-DT,T0,T0+DT,dbinder_dbeta,tol_T,&T_dbinder_dbeta_max);

   printf("T_chi_max T_Cv_max T_dm_dbeta_max T_dbinder_dbeta_max chi_max Cv_max  dm_dbeta_max dbinder_dbeta_max \n");
   printf("%11.8lf %11.8lf %11.8lf %11.8lf   %le %le %le %le\n",T_chi_max,T_Cv_max,T_dm_dbeta_max,T_dbinder_dbeta_max
                                             ,chi_max,Cv_max,dm_dbeta_max,dbinder_dbeta_max);
}
