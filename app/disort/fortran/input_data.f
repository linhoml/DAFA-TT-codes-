C----------------------------------------------------------------------C
C     This program is used to read the input data                      C
C----------------------------------------------------------------------C
C
      SUBROUTINE input(wavelen, samples, lines, n_wave, n_hours,      
     1           n_columns, height, density, f0, press, soz,press_surf,   
     2           temp, temp_surf, co2_column, co2_mixradio, dust_re,  	
	3           dust_mixradio, watice_column, watice_mixradio,   
     4           watice_re, wv_column,wv_mixradio, vz, pa, rf_ra)

      implicit none
      integer :: i, j, ios, hnum, nk
      integer :: n_hours, n_columns, n_wave, samples, lines
      real :: height(n_columns),co2_column(n_hours),f0(n_hours),
     1		soz(n_hours),press_surf(n_hours),temp_surf(n_hours),
     2	    temp(n_hours,n_columns),press(n_columns),
	3        co2_mixradio(n_hours,n_columns),
     4        density(n_hours,n_columns),dust_re(n_hours,n_columns), 
	5        dust_mixradio(n_hours,n_columns),watice_column(n_hours),
	6        watice_mixradio(n_hours,n_columns),
     7        watice_re(n_hours,n_columns),wv_column(n_hours),
     8        wv_mixradio(n_hours,n_columns)
	real :: vz(samples, lines), pa(samples, lines), wavelen(n_wave)
	real :: rf(samples, n_wave), rf_ra(samples, lines, n_wave)
      character(len=100) :: filename
      character(len=10000) :: line	  !        ݵ    ַ     
      character(len=10) :: chat
	real :: samevar

    ! ָ  Ҫ 򿪵  ļ   
      filename = 'input\CO2 column(kgm2).txt'

    !    ļ  Զ ȡ    
      open(unit=10, file=filename,status='old',action='read')

    !     ǰ   ע    
      do i = 1, 10  !     ʵ  ע         е   
        read(10, '(A)') line	   !  ȡһ    
      end do							
    !   ȡ     
 	DO i=1,n_hours
	  read(10,*) samevar, co2_column(i)  
	ENDDO

    !  ر  ļ 
      close(10)

    ! ָ  Ҫ 򿪵  ļ   
	  filename = 'input\CO2 volume mixing ratio.txt'
        open(unit=10, file=filename,status='old',action='read')

      do i = 1, 10  
        read(10, '(A)') line	   
      end do

 	DO i=1, n_columns
	  read(10,*) height(i), co2_mixradio(1,i) 
	ENDDO

      close(10)

    ! ָ  Ҫ 򿪵  ļ    
	  filename = 'input\Density(kgm3)day.txt'
        open(unit=10, file=filename,status='old',action='read')

      do i = 1, 10  
        read(10, '(A)') line	   
      end do

 	DO i=1, n_columns
	  read(10,*) samevar, density(1,i) 
	ENDDO

      close(10)

    ! ָ  Ҫ 򿪵  ļ    
	  filename = 'input\Dust effective radius(m).txt'
        open(unit=10, file=filename,status='old',action='read')

      do i = 1, 10  
        read(10, '(A)') line	   
      end do

 	DO i=1, n_columns
	  read(10,*) height(i), dust_re(1,i) 
	ENDDO

      close(10)

    ! ָ  Ҫ 򿪵  ļ    
	  filename = 'input\Dust mass mixing ratio(kgkg).txt'
        open(unit=10, file=filename,status='old',action='read')

      do i = 1, 10  
        read(10, '(A)') line	   
      end do

 	DO i=1, n_columns
	  read(10,*) height(i), dust_mixradio(1,i) 
	ENDDO

      close(10)

    ! ָ  Ҫ 򿪵  ļ    
	  filename = 'input\Pressure(Pa)0h.txt'
        open(unit=10, file=filename,status='old',action='read')
      do i = 1, 10  
        read(10, '(A)') line	   
      end do
 	DO i=1,n_columns
        read(10, '(A)') line
	  read(line,'(2E15.5)',IOSTAT=ios)  samevar, press(i) 
	ENDDO

      close(10)

    ! ָ  Ҫ 򿪵  ļ    
	  filename = 'input\Solar zenith angle(deg).txt'
        open(unit=10, file=filename,status='old',action='read')

      do i = 1, 10  
        read(10, '(A)') line	   
      end do

 	DO i=1,n_hours
	  read(10,*) samevar, soz(i) 
	ENDDO

      close(10)

    ! ָ  Ҫ 򿪵  ļ    
	  filename = 'input\Surface Pressure(Pa).txt'
        open(unit=10, file=filename,status='old',action='read')

      do i = 1, 10  
        read(10, '(A)') line	   
      end do

 	DO i=1,n_hours
	  read(10,*) samevar, press_surf(i) 
	ENDDO

      close(10)

	  filename = 'input\Surface Temperature(K)day.txt'
        open(unit=10, file=filename,status='old',action='read')

      do i = 1, 10  
        read(10, '(A)') line	   
      end do

 	DO i=1,n_hours
	  read(10,*) samevar, temp_surf(i) 
	ENDDO

      close(10)

    ! ָ  Ҫ 򿪵  ļ    
	  filename = 'input\Temperature(K)day.txt'
        open(unit=10, file=filename,status='old',action='read')

      do i = 1, 10  
        read(10, '(A)') line	   
      end do

 	DO i=1, n_columns
	  read(10,*) height(i), temp(1,i) 
	ENDDO

      close(10)


    ! ָ  Ҫ 򿪵  ļ   
      filename = 'input\Water ice column(kgm2).txt'

      open(unit=10, file=filename,status='old',action='read')

      do i = 1, 10  
        read(10, '(A)') line	 
      end do							

 	DO i=1, n_hours
	  read(10,*) height(i), watice_column(i) 
	ENDDO
 
      close(10)

    ! ָ  Ҫ 򿪵  ļ   	
	  filename = 'input\Water ice mixing ratio.txt'
        open(unit=10, file=filename,status='old',action='read')

      do i = 1, 10  
        read(10, '(A)') line	   
      end do

 	DO i=1, n_columns
	  read(10,*) height(i), watice_mixradio(1,i) 
	ENDDO

      close(10)

    ! ָ  Ҫ 򿪵  ļ   	 
	  filename = 'input\Water ice effective radius(m).txt'
        open(unit=10, file=filename,status='old',action='read')

      do i = 1, 10  
        read(10, '(A)') line	   
      end do

 	DO i=1, n_columns
	  read(10,*) height(i), watice_re(1,i) 
	ENDDO

      close(10)

    ! ָ  Ҫ 򿪵  ļ   
      filename = 'input\Water vapor column(kgm2).txt'
      open(unit=10, file=filename,status='old',action='read')

      do i = 1, 10  
        read(10, '(A)') line	 
      end do							

 	DO i=1,n_hours
	  read(10,*) samevar, wv_column(i)  
	ENDDO

      close(10)

    ! ָ  Ҫ 򿪵  ļ   
	  filename = 'input\Water vapor mixing ratio.txt'
        open(unit=10, file=filename,status='old',action='read')

      do i = 1, 10  
        read(10, '(A)') line	   
      end do

 	DO i=1, n_columns
	  read(10,*) height(i),  wv_mixradio(1,i) 
	ENDDO

      close(10)

    ! ָ  Ҫ 򿪵  ļ   
	  filename = 'input\c9dbrad2.txt'
        open(unit=10, file=filename,status='old',action='read')

 	DO j = 1, lines
	  DO i = 1, n_wave
           read(10,*) rf_ra(1:samples,j,i)
	  ENDDO
	ENDDO

      close(10)

      RETURN
      END


 

