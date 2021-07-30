/**
 * @file main.c
 * @brief This file contains the source code of the application to parallelise.
 * @details This application is a classic heat spread simulation.
 * @author Ludovic Capelli
 **/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <inttypes.h>
#include <math.h>
#include <sched.h>
#include <unistd.h>
#include <string.h>

#include "util.h"

/**
 * @argv[0] Name of the program
 * @argv[1] path to the dataset to load
 **/
int main(int argc, char* argv[])
{
	MPI_Init(NULL, NULL);

	/////////////////////////////////////////////////////
	// -- PREPARATION 1: COLLECT USEFUL INFORMATION -- //
	/////////////////////////////////////////////////////
	// Ranks for convenience so that we don't throw raw values all over the code
	const int MASTER_PROCESS_RANK = 0;

	// The rank of the MPI process in charge of this instance
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	// Number of MPI processes in total, commonly called "comm_size" for "communicator size".
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	/// Rank of the first MPI process
	const int FIRST_PROCESS_RANK = 0;
	/// Rank of the last MPI process
	const int LAST_PROCESS_RANK = comm_size - 1;

	// Rank of my up neighbour if any
	int up_neighbour_rank = (my_rank == FIRST_PROCESS_RANK) ? MPI_PROC_NULL : my_rank - 1;
	
	// Rank of my down neighbour if any
	int down_neighbour_rank = (my_rank == LAST_PROCESS_RANK) ? MPI_PROC_NULL : my_rank + 1;

	//report_placement();

	////////////////////////////////////////////////////////////////////
	// -- PREPARATION 2: INITIALISE TEMPERATURES ON MASTER PROCESS -- //
	////////////////////////////////////////////////////////////////////

	/// Array that will contain my part chunk. It will include the 2 ghost rows (1 up, 1 down)
	double temperatures[ROWS_PER_MPI_PROCESS+2][COLUMNS_PER_MPI_PROCESS];
	/// Temperatures from the previous iteration, same dimensions as the array above.
	double temperatures_last[ROWS_PER_MPI_PROCESS+2][COLUMNS_PER_MPI_PROCESS];
	/// On master process only: contains all temperatures read from input file.
	double all_temperatures[ROWS][COLUMNS];
	/// The last snapshot made JF: Moved here to pass it to the data environment.
	double snapshot[ROWS][COLUMNS];

	// The master MPI process will read a chunk from the file, send it to the corresponding MPI process and repeat until all chunks are read.
	if(my_rank == MASTER_PROCESS_RANK)
	{
		initialise_temperatures(all_temperatures);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	///////////////////////////////////////////
	//     ^                                 //
	//    / \                                //
	//   / | \    CODE FROM HERE IS TIMED    //
	//  /  o  \                              //
	// /_______\                             //
	///////////////////////////////////////////

	#pragma acc set device_num(my_rank)
        {
	#pragma acc data create(temperatures, temperatures_last, snapshot)
	{
	////////////////////////////////////////////////////////
	// -- TASK 1: DISTRIBUTE DATA TO ALL MPI PROCESSES -- //
	////////////////////////////////////////////////////////
	double total_time_so_far = 0.0;
	double start_time = MPI_Wtime();

	// JF: I let this part to run completely in the host, the loop only takes place once in the master rank
	// and copying the whole all_temperatures to device seems like a waste.
	// Q: is there a way to copy only the relevant part to master rank to its device?
	if(my_rank == MASTER_PROCESS_RANK)
	{
		for(int i = 0; i < comm_size; i++)
		{
			// Is the i'th chunk meant for me, the master MPI process?
			if(i != my_rank)
			{
				// No, so send the corresponding chunk to that MPI process.
				MPI_Send(&all_temperatures[i * ROWS_PER_MPI_PROCESS][0], ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			}
			else
			{
				// Yes, let's copy it straight for the array in which we read the file into.
				for(int j = 1; j <= ROWS_PER_MPI_PROCESS; j++)
				{
					for(int k = 0; k < COLUMNS_PER_MPI_PROCESS; k++)
					{
						temperatures_last[j][k] = all_temperatures[j-1][k];
					}
				}
			}
		}
	}
	else
	{
		// Receive my chunk.
		MPI_Recv(&temperatures_last[1][0], ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE, MASTER_PROCESS_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	// Copy the temperatures into the current iteration temperature as well
	// JF: This will be computed in the device, so we need to fetch some values from the host
	#pragma acc update device(temperatures_last)
	#pragma acc kernels
	for(int i = 1; i <= ROWS_PER_MPI_PROCESS; i++)
	{
		for(int j = 0; j < COLUMNS_PER_MPI_PROCESS; j++)
		{
			temperatures[i][j] = temperatures_last[i][j];
		}
	}

	if(my_rank == MASTER_PROCESS_RANK)
	{
		printf("Data acquisition complete.\n");
	}

	// Wait for everybody to receive their part before we can start processing
	MPI_Barrier(MPI_COMM_WORLD);
	
	/////////////////////////////
	// TASK 2: DATA PROCESSING //
	/////////////////////////////
	int iteration_count = 0;
	/// Maximum temperature change observed across all MPI processes
	double global_temperature_change;
	/// Maximum temperature change for us
	double my_temperature_change; 
	double my_temperature_change_inner, my_temperature_change_bdry;

	while(total_time_so_far < MAX_TIME)
	{
		my_temperature_change_inner = my_temperature_change_bdry = 0.0;

		// ////////////////////////////////////////
		// -- SUBTASK 1: EXCHANGE GHOST CELLS -- //
		// ////////////////////////////////////////
		
		// JF: Copy halo strips from device to update them using MPI
		#pragma acc update host(temperatures[1:1][1:COLUMNS_PER_MPI_PROCESS], temperatures[ROWS_PER_MPI_PROCESS:1][1:COLUMNS_PER_MPI_PROCESS])

		// JF: Replace two MPI calls by single Sendrecv call
		MPI_Sendrecv(&temperatures[1][0], COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE, up_neighbour_rank, 0,
				&temperatures_last[ROWS_PER_MPI_PROCESS+1][0], COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE, down_neighbour_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Sendrecv(&temperatures[ROWS_PER_MPI_PROCESS][0], COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE, down_neighbour_rank, 0,
				&temperatures_last[0][0], COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE, up_neighbour_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


		//JF: copy back updated halos to device
		#pragma acc update device(temperatures_last[0:1][1:COLUMNS_PER_MPI_PROCESS], temperatures_last[ROWS_PER_MPI_PROCESS+1:1][1:COLUMNS_PER_MPI_PROCESS])

		/////////////////////////////////////////////
		// -- SUBTASK 2: PROPAGATE TEMPERATURES -- //
		/////////////////////////////////////////////

		//JF: Main computation offloaded to device
		#pragma acc parallel loop reduction(max:my_temperature_change_bdry) async(1)
		for(int i = 1; i <= ROWS_PER_MPI_PROCESS; i++)
		{
			// Process the cell at the first column, which has no left neighbour
			if(temperatures[i][0] != MAX_TEMPERATURE)
			{
				temperatures[i][0] = (temperatures_last[i-1][0] +
									  temperatures_last[i+1][0] +
									  temperatures_last[i  ][1]) / 3.0;
				my_temperature_change_bdry = fmax(fabs(temperatures[i][0] - temperatures_last[i][0]), my_temperature_change_bdry);
			}
			// Process the cell at the last column, which has no right neighbour
			if(temperatures[i][COLUMNS_PER_MPI_PROCESS - 1] != MAX_TEMPERATURE)
			{
				temperatures[i][COLUMNS_PER_MPI_PROCESS - 1] = (temperatures_last[i-1][COLUMNS_PER_MPI_PROCESS - 1] +
														    temperatures_last[i+1][COLUMNS_PER_MPI_PROCESS - 1] +
															    temperatures_last[i  ][COLUMNS_PER_MPI_PROCESS - 2]) / 3.0;
				my_temperature_change_bdry = fmax(fabs(temperatures[i][COLUMNS_PER_MPI_PROCESS-1] - temperatures_last[i][COLUMNS_PER_MPI_PROCESS-1]), my_temperature_change_bdry);
			}
		}

		// JF: Put this calculation in separate loop so the compiler collapses it, additionally it may be 
		// executed asynchronously with the updates in the first and last column
		#pragma acc parallel loop collapse(2) reduction(max:my_temperature_change_inner) async(2)
		for(int i = 1; i <= ROWS_PER_MPI_PROCESS; i++)
		{
			// Process all cells between the first and last columns excluded, which each has both left and right neighbours
			for(int j = 1; j < COLUMNS_PER_MPI_PROCESS - 1; j++)
			{
				if(temperatures[i][j] != MAX_TEMPERATURE)
				{
					temperatures[i][j] = 0.25 * (temperatures_last[i-1][j  ] +
												 temperatures_last[i+1][j  ] +
												 temperatures_last[i  ][j-1] +
												 temperatures_last[i  ][j+1]);
					my_temperature_change_inner = fmax(fabs(temperatures[i][j] - temperatures_last[i][j]), my_temperature_change_inner);
				}
			}
		}


		///////////////////////////////////////////////////////
		// -- SUBTASK 3: CALCULATE MAX TEMPERATURE CHANGE -- //
		///////////////////////////////////////////////////////
		// JF: make sure computation in bdry and inner parts is done
		#pragma acc wait
		my_temperature_change = fmax(my_temperature_change_bdry, my_temperature_change_inner);

		//////////////////////////////////////////////////////////
		// -- SUBTASK 4: FIND MAX TEMPERATURE CHANGE OVERALL -- //
		//////////////////////////////////////////////////////////

		// JF: Let this step take place in the host and rely on MPI's allreduce instead doing it manually
		MPI_Allreduce(&my_temperature_change, &global_temperature_change, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	
		//////////////////////////////////////////////////
		// -- SUBTASK 5: UPDATE LAST ITERATION ARRAY -- //
		//////////////////////////////////////////////////
		
		// JF: Update offloaded to device
		#pragma acc kernels
		for(int i = 1; i <= ROWS_PER_MPI_PROCESS; i++)
		{
			for(int j = 0; j < COLUMNS_PER_MPI_PROCESS; j++)
			{
				temperatures_last[i][j] = temperatures[i][j];
			}
		}

		///////////////////////////////////
		// -- SUBTASK 6: GET SNAPSHOT -- //
		///////////////////////////////////
		if(iteration_count % SNAPSHOT_INTERVAL == 0)
		{
			if(my_rank == MASTER_PROCESS_RANK)
			{
				for(int j = 0; j < comm_size; j++)
				{
					if(j == my_rank)
					{
						// JF: temperatures is already in the device, we update the corresponding part
						// of snapshot there and then update the copy on the host
						#pragma acc kernels
						// Copy locally my own temperature array in the global one
						for(int k = 0; k < ROWS_PER_MPI_PROCESS; k++)
						{
							for(int l = 0; l < COLUMNS_PER_MPI_PROCESS; l++)
							{
								snapshot[j * ROWS_PER_MPI_PROCESS + k][l] = temperatures[k + 1][l];
							}
						}
						#pragma acc update host(snapshot[j * ROWS_PER_MPI_PROCESS:ROWS_PER_MPI_PROCESS][:COLUMNS_PER_MPI_PROCESS])
					}
					else
					{
						MPI_Recv(&snapshot[j * ROWS_PER_MPI_PROCESS][0], ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE, j, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					}
				}

				printf("Iteration %d: %.18f\n", iteration_count, global_temperature_change);
			}
			else
			{
				// JF: update temperatures in the host before calling MPI
				#pragma acc update host(temperatures)
				// Send my array to the master MPI process
				MPI_Send(&temperatures[1][0], ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE, MASTER_PROCESS_RANK, 0, MPI_COMM_WORLD);
			}
		}
		// Calculate the total time spent processing
		if(my_rank == MASTER_PROCESS_RANK)
		{
			total_time_so_far = MPI_Wtime() - start_time;
		}

		// Send total timer to everybody so they too can exit the loop if more than the allowed runtime has elapsed already
		MPI_Bcast(&total_time_so_far, 1, MPI_DOUBLE, MASTER_PROCESS_RANK, MPI_COMM_WORLD);

		// Update the iteration number
		iteration_count++;
	}
	///////////////////////////////////////////////
	//     ^                                     //
	//    / \                                    //
	//   / | \    CODE FROM HERE IS NOT TIMED    //
	//  /  o  \                                  //
	// /_______\                                 //
	///////////////////////////////////////////////

	/////////////////////////////////////////
	// -- FINALISATION 2: PRINT SUMMARY -- //
	/////////////////////////////////////////
	if(my_rank == MASTER_PROCESS_RANK)
	{
		printf("The program took %.2f seconds in total and executed %d iterations.\n", total_time_so_far, iteration_count);
	}
	
	} // END OF DATA REGION: #pragma acc data create(temperatures, temperatures_last)
	} // END OF SET DEVICE_NUM region

	MPI_Finalize();

	return EXIT_SUCCESS;
}
