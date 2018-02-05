#define _MAIN
#include <stdio.h>
#include <string.h>
#include <meraxes.h>
#include <sys/stat.h>
#include <mpi.h>
#include <mlog.h>
#include <math.h>

/*
 * Header info
 */

typedef struct grid_params_t {
    double SfEfficiency;
    double SfEfficiencyScaling;
    double SnEjectionEff;
    double SnEjectionScaling;
    double SnEjectionNorm;
    double SnReheatEff;
    double SnReheatLimit;
    double SnReheatScaling;
    double SnReheatNorm;
    double ReincorporationEff;
} grid_params_t;

MPI_Comm world_comm;
int world_rank;


/*
 * Code
 */

static void update_params(grid_params_t *grid_params, int i_run)
{
    physics_params_t *params = &(run_globals.params.physics);

    params->SfEfficiency        = pow(10., grid_params[i_run].SfEfficiency);
    params->SfEfficiencyScaling = grid_params[i_run].SfEfficiencyScaling + 1;
    params->SnEjectionEff       = grid_params[i_run].SnEjectionEff + 1;
    params->SnEjectionScaling   = grid_params[i_run].SnEjectionScaling + 1;
    params->SnEjectionNorm      = grid_params[i_run].SnEjectionNorm;
    params->SnReheatEff         = grid_params[i_run].SnReheatEff;
    params->SnReheatLimit       = grid_params[i_run].SnReheatLimit;
    params->SnReheatScaling     = grid_params[i_run].SnReheatScaling + 1;
    params->SnReheatNorm        = grid_params[i_run].SnReheatNorm;
    params->ReincorporationEff  = grid_params[i_run].ReincorporationEff + 1;
}

static int read_grid_params(char *fname, grid_params_t *grid_params, int n_param_combinations)
{
    int ii = 0;

    if (world_rank == 0)
    {
        FILE *fd;
        if ((fd = fopen(fname, "r")) == NULL)
        {
            fprintf(stderr, "Failed to open %s\n", fname);
            ABORT(EXIT_FAILURE);
        }

        mlog("Reading %d parameter sets...", MLOG_MESG, n_param_combinations);

        for (ii = 0; ii < n_param_combinations; ii++)
        {
            int ret = fscanf(fd, "%lg %lg %lg %lg %lg %lg %lg %lg %lg %lg",
                    &(grid_params[ii].SfEfficiency),
                    &(grid_params[ii].SfEfficiencyScaling),
                    &(grid_params[ii].SnEjectionEff),
                    &(grid_params[ii].SnEjectionScaling),
                    &(grid_params[ii].SnEjectionNorm),
                    &(grid_params[ii].SnReheatEff),
                    &(grid_params[ii].SnReheatLimit),
                    &(grid_params[ii].SnReheatScaling),
                    &(grid_params[ii].SnReheatNorm),
                    &(grid_params[ii].ReincorporationEff));
            if (ret != 10)
            {
                fprintf(stderr, "Failed to read %s correctly\n", fname);
                ABORT(EXIT_FAILURE);
            }
        }

        fclose(fd);
    }

    if (ii != n_param_combinations) {
        fprintf(stderr, "Did not successfully read %d param combinations from %s .\n", n_param_combinations, fname);
        ABORT(EXIT_FAILURE);
    }


    MPI_Bcast(grid_params, sizeof(grid_params_t)*n_param_combinations, MPI_BYTE, 0, world_comm);

    return n_param_combinations;
}


int main(int argc, char *argv[])
{

    // deal with any input arguments
    if (argc != 4)
    {
        fprintf(stderr, "\n  usage: %s <input.par> <gridfile> <n_param_combinations>\n\n", argv[0]);
        ABORT(EXIT_SUCCESS);
    }

    // init MPI
    MPI_Comm model_comm;
    int world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &world_comm);
    MPI_Comm_size(world_comm, &world_size);
    MPI_Comm_rank(world_comm, &world_rank);
    bool analysis_rank = (world_rank == world_size-1);
    int my_color = analysis_rank ? MPI_UNDEFINED : 0;
    MPI_Comm_split(world_comm, my_color, world_rank, &model_comm);
    int model_rank = -1;
    MPI_Comm_rank(model_comm, &model_rank);

    // init mlog
    if (!analysis_rank)
        init_mlog(model_comm, stdout, stdout, stderr);

    struct stat filestatus;
    int n_grid_runs;
    char cmd[STRLEN];
    char file_name_galaxies[STRLEN];
    char output_dir[STRLEN];

    // read in the grid parameters
    int n_param_combinations = atoi(argv[3]);
    grid_params_t *grid_params = malloc(sizeof(grid_params_t) * n_param_combinations);
    MPI_Bcast(&n_param_combinations, 1, MPI_INT, 0, world_comm);
    if (world_rank > 0)
        grid_params = malloc(sizeof(grid_params_t) * n_param_combinations);
    n_grid_runs = read_grid_params(argv[2], grid_params, n_param_combinations);

    if (!analysis_rank)
    {
        // debugging
#ifdef DEBUG
        if (model_rank == 0)
            mpi_debug_here();
#endif

        // read the input parameter file
        read_parameter_file(argv[1], 0);

        // set the interactive flag
        run_globals.params.FlagInteractive = 1;

        // Check to see if the output directory exists and if not, create it
        if (stat(run_globals.params.OutputDir, &filestatus) != 0)
            mkdir(run_globals.params.OutputDir, 02755);

        // initiate meraxes
        init_meraxes();

        // calculate the output hdf5 file properties for later use
        calc_hdf5_props();
    }

    // Copy the output dir of the model to the analysis rank
    if (world_rank == 0)
        MPI_Send(run_globals.params.OutputDir, STRLEN, MPI_CHAR, world_size-1, 16, world_comm);
    if (analysis_rank)
        MPI_Recv(output_dir, STRLEN, MPI_CHAR, 0, 16, world_comm, MPI_STATUS_IGNORE);

    // Run the model!
    for (int ii = 0; ii < n_grid_runs; ii++)
    {
        sprintf(file_name_galaxies, "meraxes_%03d", ii);
        if (!analysis_rank)
        {
            strcpy(run_globals.params.FileNameGalaxies, file_name_galaxies);
            mlog(">>>> New FileNameGalaxies is: %s", MLOG_MESG, run_globals.params.FileNameGalaxies);
            update_params(grid_params, ii);

            mlog(">>>> Updated params to: %g %g %g %g %g %g %g %g %g %g", MLOG_MESG,
                    grid_params[ii].SfEfficiency,
                    grid_params[ii].SfEfficiencyScaling,
                    grid_params[ii].SnEjectionEff,
                    grid_params[ii].SnEjectionScaling,
                    grid_params[ii].SnEjectionNorm,
                    grid_params[ii].SnReheatEff,
                    grid_params[ii].SnReheatLimit,
                    grid_params[ii].SnReheatScaling,
                    grid_params[ii].SnReheatNorm,
                    grid_params[ii].ReincorporationEff);

            dracarys();
        }

        // Sync here so that we can ensure we are not going to try and read
        // non-existent files
        MPI_Barrier(world_comm);

        if (analysis_rank)
        {
            // N.B. The script called here should delete the output files once it is finished with them
            sprintf(cmd, "/home/smutch/miniconda3/bin/python analyse_run.py %s/%s.hdf5",
                    output_dir, file_name_galaxies);
            printf(" >>>> ANALYSIS : Calling\n\t%s\n", cmd);
            system(cmd);
        }
    }

    // cleanup
    if (!analysis_rank)
        cleanup();

    free(grid_params);
    MPI_Comm_free(&world_comm);

    // Only the analysis rank will make it here
    MPI_Finalize();
    exit(EXIT_SUCCESS);
}
