C----------------------------------------------------------------------C
C     This program is used to read the optical data of gas and aerosol C
C----------------------------------------------------------------------C
C
      SUBROUTINE optical_data(n_wave, wavnum_co2, sw_co2,  
     1           gammaa_co2, gammas_co2, elower_co2,nn_co2, delta_co2,
     2	       ext_dust, ww_dust, g_dust,
	3		   wavnum_h2o, sw_h2o, gammaa_h2o, gammas_h2o,
	4		   elower_h2o,nn_h2o, delta_h2o,
	5           ext_watice, ww_watice, g_watice)

      implicit none
      integer :: i, ios, hnum
      integer :: n_wave
	real :: wavnum_co2(n_wave), sw_co2(n_wave), gammaa_co2(n_wave), 
     1        gammas_co2(n_wave), elower_co2(n_wave),nn_co2(n_wave),
	2        delta_co2(n_wave)
	real :: wavnum_h2o(n_wave), sw_h2o(n_wave), gammaa_h2o(n_wave), 
     1        gammas_h2o(n_wave), elower_h2o(n_wave),nn_h2o(n_wave),
	2        delta_h2o(n_wave)
      real :: wavl, ext_dust(n_wave), ww_dust(n_wave), g_dust(n_wave),
     1        ext_watice(n_wave), ww_watice(n_wave), g_watice(n_wave)
	real :: wavenum
      character(len=100) :: filename

    ! read co2 optical propertise
      filename = 'optical\co2_hitran.txt'
      open(unit=10, file=filename,status='old',action='read')
						
 	DO i=1,n_wave
	     read(10,'(F12.6,F12.6,E10.3,F5.4,F5.3,F10.4,F4.2,F8.6)',
     1         IOSTAT=ios) wavenum, wavnum_co2(i), sw_co2(i), 
     2         gammaa_co2(i), gammas_co2(i), elower_co2(i), 
     3         nn_co2(i), delta_co2(i) 
	ENDDO
      close(10)

    ! read h2o optical propertise
      filename = 'optical\h2o_hitran.txt'
      open(unit=10, file=filename,status='old',action='read')
						
 	DO i=1,n_wave
	     read(10,'(F12.6,F12.6,E10.3,F5.4,F5.3,F10.4,F4.2,F8.6)',
     1         IOSTAT=ios) wavenum, wavnum_h2o(i), sw_h2o(i), 
     2         gammaa_h2o(i), gammas_h2o(i), elower_h2o(i), 
     3         nn_h2o(i), delta_h2o(i) 
	ENDDO
      close(10)


    ! read dust optical propertise
      filename = 'optical\mie_dust.dat'
      open(unit=10, file=filename,status='old',action='read')
						
 	DO i=1,n_wave
        read(10, *) wavl, ext_dust(i), ww_dust(i), g_dust(i)
	ENDDO

      close(10)

    ! read ice water optical propertise	
      filename = 'optical\mie_icewater.dat'
      open(unit=10, file=filename,status='old',action='read')	
					
 	DO i=1,n_wave
        read(10, *) wavl, ext_watice(i), ww_watice(i), g_watice(i)
	ENDDO

      close(10)

      RETURN
      END


 

