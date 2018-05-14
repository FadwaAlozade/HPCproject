#include <stdio.h>
#include <math.h>
#include <shalw.h>
#include <export.h>
#include <immintrin.h>

const __m256d salpha = _mm256_set1_pd(alpha);
const __m256d sdissip = _mm256_set1_pd(dissip);
const __m256d spcor4 = _mm256_set1_pd(pcor/4.);
const __m256d sgrav = _mm256_set1_pd(-grav);
const __m256d shmoy = _mm256_set1_pd(hmoy);
const __m256d s1dx = _mm256_set1_pd(1./dx);
const __m256d s1dy = _mm256_set1_pd(1./dy);
const __m256d sdt = _mm256_set1_pd(dt);
const __m256d s2 = _mm256_set1_pd(-2);

//__m256 result = _mm_mul_ps(vector, scalar);

inline __m256d hFil_forward(int t, int i, int j) {
  //Phase d'initialisation du filtre
  //HPHY(t - 1, i, j) est encore nul
	__m256d vhphy, vhphyt, vres;
	if (t <= 2){
		vres = _mm256_load_pd(&HPHY(t, i, j));
	}
	else {
		vhphy = _mm256_load_pd(&HPHY(t - 1, i, j));
		vhphyt = _mm256_load_pd(&HPHY(t, i, j));

		vres = _mm256_mul_pd(vhphy, s2);
		vres = _mm256_add_pd(vres, vhphy);
		vres = _mm256_add_pd(vres, vhphyt);
		vres = _mm256_mul_pd(vres, salpha);
		vres = _mm256_add_pd(vres, vhphy);
	}
	
	return vres;
	//return HPHY(t - 1, i, j) +  alpha * (HFIL(t - 1, i, j) - 2 * HPHY(t - 1, i, j) + HPHY(t, i, j));
}

inline __m256d uFil_forward(int t, int i, int j) {
	//Phase d'initialisation du filtre
	//UPHY(t - 1, i, j) est encore nul
	__m256d vuphy, vuphyt, vufil, vres;

	if (t <= 2) vres = _mm256_load_pd(&UPHY(t, i, j));
	else {
		vuphy = _mm256_load_pd(&UPHY(t - 1, i, j));
		vuphyt = _mm256_load_pd(&UPHY(t, i, j));
		vufil = _mm256_load_pd(&UFIL(t - 1, i, j));

		vres = _mm256_mul_pd(vuphy, s2);
		vres = _mm256_add_pd(vres, vufil);
		vres = _mm256_add_pd(vres, vuphyt);
		vres = _mm256_mul_pd(vres, salpha);
		vres = _mm256_add_pd(vres, vuphy);
	}
	return vres;
	//return UPHY(t - 1, i, j) + alpha * (UFIL(t - 1, i, j) - 2 * UPHY(t - 1, i, j) + UPHY(t, i, j));
}

inline __m256d vFil_forward(int t, int i, int j) {
	//Phase d'initialisation du filtre
	//VPHY(t - 1, i, j) est encore nul
	__m256d vvphy, vvphyt, vvfil, vres;
	if (t <= 2) vres = _mm256_load_pd(&VPHY(t, i, j));
	else {
		vvphy = _mm256_load_pd(&VPHY(t - 1, i, j));
		vvphyt = _mm256_load_pd(&VPHY(t, i, j));
		vvfil = _mm256_load_pd(&VFIL(t - 1, i, j));

		vres = _mm256_mul_pd(vvphy, s2);
		vres = _mm256_add_pd(vres, vvfil);
		vres = _mm256_add_pd(vres, vvphyt);
		vres = _mm256_mul_pd(vres, salpha);
		vres = _mm256_add_pd(vres, vvphy);
	}
	return vres;
  //return VPHY(t - 1, i, j) + alpha * (VFIL(t - 1, i, j) - 2 * VPHY(t - 1, i, j) + VPHY(t, i, j));
}

inline __m256d hPhy_forward(int t, int i, int j) {
	__m256d vuphy, vvphy, vhfil, vres, vres2;
	vuphy = _mm256_load_pd(&UPHY(t - 1, i, j));
	vvphy = _mm256_load_pd(&VPHY(t - 1, i, j));
	vhfil = _mm256_load_pd(&HFIL(t - 1, i, j));

	__m256d vc = _mm256_setzero_pd();
	__m256d vd = _mm256_setzero_pd();

	if (i > 0)
		vc = _mm256_load_pd(&UPHY(t - 1, i - 1, j));

	if (j < size_y - 1)
		vd = _mm256_load_pd(&VPHY(t - 1, i, j + 1));

	vres = _mm256_sub_pd(vuphy, vc);
	vres = _mm256_mul_pd(vres, s1dx);

	vres2 = _mm256_sub_pd(vd, vvphy);
	vres2 = _mm256_mul_pd(vres2, s1dy);

	vres = _mm256_add_pd(vres, vres2);

	vres = _mm256_mul_pd(sdt, vres);
	vres = _mm256_mul_pd(shmoy, vres);

	return vres;

	//return HFIL(t - 1, i, j) -	dt * hmoy * ((UPHY(t - 1, i, j) - c) / dx +	 	(d - VPHY(t - 1, i, j)) / dy);
}

inline __m256d uPhy_forward(int t, int i, int j) {
  __m256d vvphy, vufil, vhphy, vres, vres1, vres2;

  vufil = _mm256_load_pd(&UFIL(t - 1, i, j));
  vvphy = _mm256_load_pd(&VPHY(t - 1, i, j));
  vhphy = _mm256_load_pd(&HPHY(t - 1, i, j));

  __m256d vb = _mm256_setzero_pd();
  __m256d ve = _mm256_setzero_pd();
  __m256d vf = _mm256_setzero_pd();
  __m256d vg = _mm256_setzero_pd();
  
  if (i == size_x - 1)
    return vb;

  
  if (i < size_x - 1)
    vb = _mm256_load_pd(&HPHY(t - 1, i + 1, j));

 
  if (j < size_y - 1)
    ve = _mm256_load_pd(&VPHY(t - 1, i, j + 1));

 
  if (i < size_x - 1)
    vf = _mm256_load_pd(&VPHY(t - 1, i + 1, j));

  
  if (i < size_x - 1 && j < size_y - 1)
    vg = _mm256_load_pd(&VPHY(t - 1, i + 1, j + 1));

	vres = _mm256_mul_pd(sgrav,s1dx);
	vres1 = _mm256_sub_pd(vb,vhphy);
	vres= _mm256_mul_pd(vres,vres1);
	vres= _mm256_mul_pd(sdt,vres);
	vres1= _mm256_add_pd(vvphy,ve);
	vres1= _mm256_add_pd(vres1,vf);
	vres1= _mm256_add_pd(vres1,vg);
	vres1=_mm256_mul_pd(spcor4,vres1);
	vres2=_mm256_mul_pd(sdissip,vufil);
	vres=_mm256_add_pd(vufil,vres);
	vres=_mm256_add_pd(vres,vres1);
	vres=_mm256_sub_pd(vres,vres2);

	return vres;

  //return UFIL(t - 1, i, j) +dt * ((-grav / dx) * (b - HPHY(t - 1, i, j)) +(pcor / 4.) * (VPHY(t - 1, i, j) + e + f + g) -(dissip * UFIL(t - 1, i, j)));
}

inline __m256d vPhy_forward(int t, int i, int j) {
  __m256d vvfil, vuphy, vhphy, vres, vres1, vres2;

  vuphy = _mm256_load_pd(&UPHY(t - 1, i, j));
  vvfil = _mm256_load_pd(&VFIL(t - 1, i, j));
  vhphy = _mm256_load_pd(&HPHY(t - 1, i, j));

  __m256d vc = _mm256_setzero_pd();
  __m256d ve = _mm256_setzero_pd();
  __m256d vf = _mm256_setzero_pd();
  __m256d vd = _mm256_setzero_pd();

  if (j == 0)
    return vc;


  if (j > 0)
    vc =_mm256_load_pd(&HPHY(t - 1, i, j - 1));


  if (i > 0 && j > 0)
    vd = _mm256_load_pd(&UPHY(t - 1, i -1, j -1));


  if (i > 0)
    ve = _mm256_load_pd(&UPHY(t - 1, i - 1, j));

  
  if (j > 0)
    vf = _mm256_load_pd(&UPHY(t - 1, i, j - 1));

	vres = _mm256_mul_pd(sgrav,s1dy);
	vres1 = _mm256_sub_pd(vhphy,vc);
	vres= _mm256_mul_pd(vres,vres1);
	vres= _mm256_mul_pd(sdt,vres);
	vres1= _mm256_add_pd(vuphy,ve);
	vres1= _mm256_add_pd(vres1,vf);
	vres1= _mm256_add_pd(vres1,vd);
	vres1=_mm256_mul_pd(spcor4,vres1);
	vres2=_mm256_mul_pd(sdissip,vvfil);
	vres=_mm256_add_pd(vvfil,vres);
	vres=_mm256_sub_pd(vres,vres1);
	vres=_mm256_sub_pd(vres,vres2);

  //return UFIL(t - 1, i, j) +dt * ((-grav / dx) * (b - HPHY(t - 1, i, j)) +(pcor / 4.) * (VPHY(t - 1, i, j) + e + f + g) -(dissip * UFIL(t - 1, i, j)));
  //return VFIL(t - 1, i, j) +  dt * ((-grav / dy) * (HPHY(t - 1, i, j) - c) - (pcor / 4.) * (d + e + f + UPHY(t - 1, i, j)) - (dissip * VFIL(t - 1, i, j)));
	return vres;
}

void forward(void) {
  FILE *file = NULL;
  double svdt = 0.;
  int t = 0;
  __m256d vhfil, vufil, vvfil, vhphy, vuphy, vvphy;
  
  if (file_export) {
    file = create_file();
    export_step(file, t);
  }
    
  for (t = 1; t < nb_steps; t++) {
    if (t == 1) {
      svdt = dt;
      dt = 0;
    }
    if (t == 2){
      dt = svdt / 2.;
    }

	for (int i = 0; i < size_x; i++) {
	    for (int j = 0; j < size_y/256; j++) {    

			// vhphy = hPhy_forward(t, i, j*8);
			// _mm256_store_ps(&HPHY(t, i, j*8), vhphy);
			
			vhphy =  hPhy_forward(t, i, j*256);
			_mm256_store_pd(&HPHY(t, i, j*256), vhphy);

			vuphy =  uPhy_forward(t, i, j*256);
			_mm256_store_pd(&UPHY(t, i, j*256), vuphy);

			vvphy =  vPhy_forward(t, i, j*256);
			_mm256_store_pd(&VPHY(t, i, j*256), vvphy);

			vhfil = hFil_forward(t, i, j*256);
			_mm256_store_pd(&HFIL(t, i, j*256) , vhfil);

			vufil = uFil_forward(t, i, j*256);
			_mm256_store_pd(&UFIL(t, i, j*256) , vufil);

			vvfil = vFil_forward(t, i, j*256);
			_mm256_store_pd(&VFIL(t, i, j*256) , vvfil);
      }
    }

    if (file_export) {
      export_step(file, t);
    }
    
    if (t == 2) {
      dt = svdt;
    }
  }

  if (file_export) {
    finalize_export(file);
  }
}
