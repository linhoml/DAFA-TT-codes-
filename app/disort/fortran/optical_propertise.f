C----------------------------------------------------------------------C
C     This program is used to calulate the optical propertise of gases C     
C----------------------------------------------------------------------C
C
      SUBROUTINE optical_calculate(n_wave, n_columns, n_hours, wavelen,    
     1           press, temp, wavnum_co2, sw_co2, gammaa_co2,gammas_co2, 
     2           elower_co2, nn_co2, delta_co2, co2_mixradio,
	3		   wavnum_h2o, sw_h2o, gammaa_h2o, gammas_h2o, 
     4           elower_h2o, nn_h2o, delta_h2o, wv_mixradio,
     5           kab_co2, kab_h2o)

      implicit none
      integer :: i, j
      integer :: n_wave, n_columns, n_hours
	real :: Tref, Qref, afaD, kk, cc, c2, Na
	real :: exE1, exE2, exv1, exv2, wavenum, waveij
	real :: wavelen(n_wave) 
	real :: wavnum_co2(n_wave), sw_co2(n_wave), gammaa_co2(n_wave), 
     1        gammas_co2(n_wave), elower_co2(n_wave), nn_co2(n_wave),
	2        delta_co2(n_wave)
	real :: wavnum_h2o(n_wave), sw_h2o(n_wave), gammaa_h2o(n_wave), 
     1        gammas_h2o(n_wave), elower_h2o(n_wave),nn_h2o(n_wave),
	2        delta_h2o(n_wave)
	real :: press(n_columns), temp(n_hours,n_columns) 
	real :: co2_mixradio(n_hours,n_columns)
	real :: wv_mixradio(n_hours,n_columns)
	real :: qij_co2(n_columns), gamma_co2(n_wave, n_columns),
     1        qij_h2o(n_columns), gamma_h2o(n_wave, n_columns)
	real :: Sij_co2(n_wave, n_columns), fij_co2(n_wave, n_columns),
     1		Sij_h2o(n_wave, n_columns), fij_h2o(n_wave, n_columns)
	real :: fijL_co2(n_wave, n_columns), fijD_co2(n_wave, n_columns)
	real :: fijL_h2o(n_wave, n_columns), fijD_h2o(n_wave, n_columns)
	real :: kab_co2(n_wave, n_columns), kab_h2o(n_wave, n_columns)

	kk = 1.38064*10E-16
	cc = 2.99792458*10E10
      c2 = 1.4387769
	Na = 6.02214129*10E23
	Tref = 296.0
	Qref = 282029.9106	!co2

    ! calculate absorption coefficient of co2
	qij_co2(:) = 0.0
	CALL qe_co2(n_columns, n_hours, temp, qij_co2)

	DO i = 1, n_wave
	    wavenum = 10000. / wavelen(i)

	    DO j = 1, n_columns
		 waveij = wavnum_co2(i) + delta_co2(i) * press(j)/101325.
    ! calculate line intensity
	     exE1 = exp(-c2*elower_co2(i)/temp(1,j))
		 exE2 = exp(-c2*elower_co2(i)/Tref)
	     exv1 = 1.0 - exp(-c2*waveij/temp(1,j))
	     exv2 = 1.0 - exp(-c2*waveij/Tref)
	     Sij_co2(i,j) = sw_co2(i) * Qref *	exE1 * exv1 / (qij_co2(j)*
     1                     exE2 * exv2)

    ! calculate line shape
	     afaD = waveij / cc * sqrt(2.* Na * kk * temp(1,j) *   
     1            log(2.) / 44.01)
		 gamma_co2(i,j) = (Tref/temp(1,j))**nn_co2(i) * 
     1                    (gammaa_co2(i) * (1.0 - co2_mixradio(1,j)) * 
     2                    press(j)/101325. + gammas_co2(i) * 
     3                    co2_mixradio(1,j) * press(j)/101325.)

           fijL_co2(i,j) = 0.318 * gamma_co2(i,j) / (gamma_co2(i,j)**2 + 
     1                     (wavenum - (waveij + delta_co2(i)*press(j)
     2                     /101325.))**2)
	     fijD_co2(i,j) =  sqrt(log(2./(3.14159*afaD**2))) * exp(-1. *
     1                  (wavenum - waveij)**2*log(2.)/(afaD**2))


	     if(press(j) .gt. 13.33) then
	       fij_co2(i,j) = fijL_co2(i,j)
		 else
	       fij_co2(i,j) = fijD_co2(i,j)	
		 endif	  
           kab_co2(i,j) = Sij_co2(i,j) * fij_co2(i,j) 
	    enddo
	enddo

	Qref = 206297.69	!h2o
    ! calculate absorption coefficient of h2o
	qij_h2o(:) = 0.0
	CALL qe_h2o(n_columns, n_hours, temp, qij_h2o)

	DO i = 1, n_wave
		wavenum = 10000. / wavelen(i)

	    DO j = 1, n_columns

 		 waveij = wavnum_co2(i) + delta_co2(i) * press(j)/101325.
    ! calculate line intensity
	     exE1 = exp(-c2*elower_co2(i)/temp(1,j))
		 exE2 = exp(-c2*elower_co2(i)/Tref)
	     exv1 = 1.0 - exp(-c2*waveij/temp(1,j))
	     exv2 = 1.0 - exp(-c2*waveij/Tref)
	     Sij_h2o(i,j) = sw_h2o(i) * Qref *	exE1 * exv1 / (qij_h2o(j)*
     1                     exE2 * exv2)

    ! calculate line shape
	     afaD = waveij / cc * sqrt(2.* Na * kk * temp(1,j) *   
     1            log(2.) / 18.)
		 gamma_h2o(i,j) = (Tref/temp(1,j))**nn_h2o(i) * 
     1                    (gammaa_h2o(i) * (1.0 - wv_mixradio(1,j)) * 
     2                    press(j)/101325. + gammas_h2o(i) * 
     3                    wv_mixradio(1,j) * press(j)/101325.)
           fijL_h2o(i,j) = 0.318 * gamma_h2o(i,j) / (gamma_h2o(i,j)**2 + 
     1                     (wavenum - (waveij + delta_h2o(i)*press(j)
     2                     /101325.))**2)
	     fijD_h2o(i,j) =  sqrt(log(2./(3.14159*afaD**2))) * exp(-1. *
     1                  (wavenum - waveij)**2*log(2.)/(afaD**2))


	     if(press(j) .gt. 13.33) then
	       fij_h2o(i,j) = fijL_h2o(i,j)
		 else
	       fij_h2o(i,j) = fijD_h2o(i,j)	
		 endif	  
           kab_h2o(i,j) = Sij_h2o(i,j) * fij_h2o(i,j) 
c	print *, wavelen(i), j, kab_co2(i,j),kab_h2o(i,j)
	    enddo
	enddo

      RETURN
      END

	SUBROUTINE qe_co2(n_columns, n_hours, temp, qij_co2)
      implicit none
      integer :: i, j, index
      integer :: n_columns, n_hours
	real :: tp(500), qq(500), closest_t, closest_q
 	real :: temp(n_hours,n_columns), qij_co2(n_columns)

	open(unit=10,file='optical\Qt_co2.txt',status='old',action='read')

	DO i = 1, 301
	   read(10,*) tp(i), qq(i)
	ENDDO
	close(10)

	DO j = 1, n_columns
 	  index = 1
	  DO i = 1, 301
	     if(abs(tp(i)-temp(1,j)) < 0.5) then
	        index = i
	     endif
	   ENDDO
	   qij_co2(j) = qq(index)
      ENDDO

      RETURN
      END

	SUBROUTINE qe_h2o(n_columns, n_hours, temp, qij_h2o)
      implicit none
      integer :: i, j, index
      integer :: n_columns, n_hours
	real :: tp(500), qq(500), closest_t, closest_q
 	real :: temp(n_hours,n_columns), qij_h2o(n_columns)

	open(unit=10,file='optical\Qt_h2o.txt',status='old',action='read')

	DO i = 1, 301
	   read(10,*) tp(i), qq(i)
	ENDDO
	close(10)

	DO j = 1, n_columns
 	  index = 1
	  DO i = 1, 301
	     if(abs(tp(i)-temp(1,j)) < 0.5) then
	        index = i
	     endif
	   ENDDO
	   qij_h2o(j) = qq(index)
      ENDDO

      RETURN
      END
