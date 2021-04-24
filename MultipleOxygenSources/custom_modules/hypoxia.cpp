/*
###############################################################################
# If you use PhysiCell in your project, please cite PhysiCell and the version #
# number, such as below:                                                      #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version 1.2.2) [1].    #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 2017 (in review).                       #
#     preprint DOI: 10.1101/088773                                            #
#                                                                             #
# Because PhysiCell extensively uses BioFVM, we suggest you also cite BioFVM  #
#     as below:                                                               #
#                                                                             #
# We implemented and solved the model using PhysiCell (Version 1.2.2) [1],    #
# with BioFVM [2] to solve the transport equations.                           #
#                                                                             #
# [1] A Ghaffarizadeh, R Heiland, SH Friedman, SM Mumenthaler, and P Macklin, #
#     PhysiCell: an Open Source Physics-Based Cell Simulator for Multicellu-  #
#     lar Systems, PLoS Comput. Biol. 2017 (in review).                       #
#     preprint DOI: 10.1101/088773                                            #
#                                                                             #
# [2] A Ghaffarizadeh, SH Friedman, and P Macklin, BioFVM: an efficient para- #
#    llelized diffusive transport solver for 3-D biological simulations,      #
#    Bioinformatics 32(8): 1256-8, 2016. DOI: 10.1093/bioinformatics/btv730   #
#                                                                             #
###############################################################################
#                                                                             #
# BSD 3-Clause License (see https://opensource.org/licenses/BSD-3-Clause)     #
#                                                                             #
# Copyright (c) 2015-2017, Paul Macklin and the PhysiCell Project             #
# All rights reserved.                                                        #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
# this list of conditions and the following disclaimer.                       #
#                                                                             #
# 2. Redistributions in binary form must reproduce the above copyright        #
# notice, this list of conditions and the following disclaimer in the         #
# documentation and/or other materials provided with the distribution.        #
#                                                                             #
# 3. Neither the name of the copyright holder nor the names of its            #
# contributors may be used to endorse or promote products derived from this   #
# software without specific prior written permission.                         #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" #
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE   #
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE  #
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE   #
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR         #
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF        #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS    #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN     #
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)     #
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE  #
# POSSIBILITY OF SUCH DAMAGE.                                                 #
#                                                                             #
###############################################################################
*/

#include "./hypoxia.h"

Cell_Definition blood_vessel_section;

void create_blood_vessel_section( void )
{
	blood_vessel_section = cell_defaults;

	blood_vessel_section.name = "blood vessel section";
	blood_vessel_section.type = 1;
	blood_vessel_section.phenotype.geometry.radius = 2*cell_defaults.phenotype.geometry.radius;
	blood_vessel_section.phenotype.volume.total = pow(blood_vessel_section.phenotype.geometry.radius,3)*4.188790204786391;

	// turn off proliferation;
	blood_vessel_section.phenotype.cycle.data.transition_rate(0,1) = 0.0;


	// set default uptake and secretion
	static int oxygen_ID = microenvironment.find_density_index( "oxygen" ); // 0

	// oxygen
	blood_vessel_section.phenotype.secretion.secretion_rates[oxygen_ID] = default_microenvironment_options.Dirichlet_condition_vector[0];
	blood_vessel_section.phenotype.secretion.uptake_rates[oxygen_ID] = 0.0;
	blood_vessel_section.phenotype.secretion.saturation_densities[oxygen_ID] = default_microenvironment_options.Dirichlet_condition_vector[0];

	// turn off apoptosis
	int apoptosis_index = blood_vessel_section.phenotype.death.find_death_model_index( PhysiCell_constants::apoptosis_death_model );
	blood_vessel_section.phenotype.death.rates[apoptosis_index] = 0;


	// set functions

	blood_vessel_section.functions.update_phenotype = NULL;
	blood_vessel_section.functions.volume_update_function = NULL;
	blood_vessel_section.functions.custom_cell_rule = NULL;
	//blood_vessel_section.functions.custom_cell_rule = NULL;
	blood_vessel_section.functions.update_migration_bias = NULL;
	blood_vessel_section.functions.update_velocity = NULL;

	//set mechanics aspects
	blood_vessel_section.phenotype.motility.is_motile = false;
	return;
}

void create_cell_types( void )
{
	SeedRandom(0);

	int oxygen_i = get_default_microenvironment()->find_density_index( "oxygen" );

	initialize_default_cell_definition();
	cell_defaults.phenotype.secretion.sync_to_microenvironment( &microenvironment );

    // cell cycle model
	cell_defaults.phenotype.cycle.sync_to_cycle_model( Ki67_basic );

	// make sure we're ready for 2D
    if( default_microenvironment_options.simulate_2D == true ){
        cell_defaults.functions.set_orientation = up_orientation;
        cell_defaults.phenotype.geometry.polarity = 1.0;
        cell_defaults.phenotype.motility.restrict_to_2D = true;
    }

	int apoptosis_index = cell_defaults.phenotype.death.find_death_model_index( PhysiCell_constants::apoptosis_death_model );

	cell_defaults.parameters.o2_proliferation_saturation = parameters.doubles["sigma_S"].value;
	cell_defaults.parameters.o2_reference = cell_defaults.parameters.o2_proliferation_saturation;
	cell_defaults.parameters.o2_proliferation_threshold = parameters.doubles["sigma_T"].value;
	cell_defaults.parameters.o2_necrosis_threshold = cell_defaults.parameters.o2_proliferation_threshold;
	cell_defaults.parameters.o2_necrosis_max = cell_defaults.parameters.o2_proliferation_threshold;

	cell_defaults.phenotype.cycle.data.transition_rate(0,1) = parameters.doubles["rate_KnToKp"].value;
	cell_defaults.phenotype.cycle.data.transition_rate(1,0) = parameters.doubles["rate_KpToKn"].value;

	// set default motiltiy
	cell_defaults.phenotype.motility.is_motile = true;
	cell_defaults.functions.update_migration_bias = oxygen_taxis_motility;

	cell_defaults.phenotype.motility.persistence_time = parameters.doubles["pers_time_red"].value;
	cell_defaults.phenotype.motility.migration_bias = parameters.doubles["bias_red"].value;
	cell_defaults.phenotype.motility.migration_speed = parameters.doubles["speed_red"].value;

	// set default uptake and secretion
	// oxygen
	cell_defaults.phenotype.secretion.secretion_rates[oxygen_i] = 0.0;
	cell_defaults.phenotype.secretion.uptake_rates[oxygen_i] = parameters.doubles["cell_oxy_cons"].value;


	cell_defaults.functions.update_phenotype = tumor_cell_phenotype;

	cell_defaults.name = "cancer cell";
	cell_defaults.type = 0;

    // turn off BM forces
    cell_defaults.phenotype.mechanics.cell_BM_adhesion_strength = 0.0;
    cell_defaults.phenotype.mechanics.cell_BM_repulsion_strength = 0.0;

	// add custom data
	std::vector<double> genes = { 1.0, 0.0 }; // RFP, GFP
	std::vector<double> proteins = {1.0, 0.0 }; // RFP, GFP;

	double default_degradation_rate = parameters.doubles["protein_deg_rate"].value;

	std::vector<double> degradation_rates = { default_degradation_rate , default_degradation_rate };

	double default_production_rate = parameters.doubles["protein_prod_rate"].value;

	std::vector<double> creation_rates = { default_production_rate , default_production_rate };

	cell_defaults.custom_data.add_vector_variable( "genes" , "dimensionless", genes );

	cell_defaults.custom_data.add_vector_variable( "proteins" , "dimensionless", proteins );
	cell_defaults.custom_data.add_vector_variable( "creation_rates" , "1/min" , creation_rates );
	cell_defaults.custom_data.add_vector_variable( "degradation_rates" , "1/min" , degradation_rates );

	cell_defaults.custom_data.add_variable( "persistence time" , "dimensionless" , 0.0 );

    std::vector<double> color = {255, 255, 255};
	cell_defaults.custom_data.add_vector_variable( "nuclear_color" , "dimensionless", color );
	cell_defaults.custom_data.add_vector_variable( "cytoplasmic_color" , "dimensionless", color );

	// create the blood_vessel_section
	create_blood_vessel_section();

	return;
}

void setup_microenvironment( void )
{
	initialize_microenvironment();

	// std::vector< double > position = {0.0,0.0,0.0};
	// for( unsigned int n=0; n < microenvironment.number_of_voxels() ; n++ ){
	// 	if (dist(microenvironment.mesh.voxels[n].center,position) >= parameters.doubles["initial_tumor_rad"].value)
	// 		microenvironment.add_dirichlet_node(n,default_microenvironment_options.Dirichlet_condition_vector);
	// }
	return;
}

std::vector<std::vector<double>> create_cell_sphere_positions(double cell_radius, double sphere_radius)
{
	std::vector<std::vector<double>> cells;
	int xc=0,yc=0,zc=0;
	double x_spacing= cell_radius*sqrt(3);
	double y_spacing= cell_radius*2;
	double z_spacing= cell_radius*sqrt(3);

	std::vector<double> tempPoint(3,0.0);

	for(double z=-sphere_radius;z<sphere_radius;z+=z_spacing, zc++)
	{
		for(double x=-sphere_radius;x<sphere_radius;x+=x_spacing, xc++)
		{
			for(double y=-sphere_radius;y<sphere_radius;y+=y_spacing, yc++)
			{
				tempPoint[0]=x + (zc%2) * 0.5 * cell_radius;
				tempPoint[1]=y + (xc%2) * cell_radius;
				tempPoint[2]=z;

				if(sqrt(norm_squared(tempPoint))< sphere_radius)
				{ cells.push_back(tempPoint); }
			}

		}
	}
	return cells;
}

void introduce_blood_vessel_sections( void )
{

	double radius = 100.0;
	std::vector<double> PosCenter = {750.0, 750.0, 0.0};

	generateSection( radius, PosCenter);

	radius = 50.0; PosCenter = {-750.0, -750.0, 0.0};
	generateSection( radius, PosCenter);
	return;
}

// bool check_neigboors()
// {
//
// }

void generateSection(double radius, std::vector<double>& PosCenter)
{
	double theta = 0;
	int number_of_elements = 0.4*radius;
	const double PI = 3.1415926535897932384626433832795;

	for( int i=0 ;i <= number_of_elements; i++ )
	{
		Cell* pCell = create_cell( blood_vessel_section );
		pCell->assign_position( radius*cos(theta) + PosCenter[0], radius*sin(theta)+ PosCenter[1], 0 );
		theta = theta + (360.0/number_of_elements)*(PI/180);
	}
	return;
}

void setup_tissue( void )
{
	static int genes_i = 0;
	static int proteins_i =1;
	static int creation_rates_i = 2;
	static int degradation_rates_i = 3;

	static int red_i = 0;
	static int green_i = 1;

	// place a cluster of tumor cells at the center
	double cell_radius = cell_defaults.phenotype.geometry.radius;
	double cell_spacing = 0.95 * 2.0 * cell_radius;

	double tumor_radius = parameters.doubles["initial_tumor_rad"].value;

	Cell* pCell = NULL;

	double x = 0.0;
	double x_outer = tumor_radius;
	double y = 0.0;

	introduce_blood_vessel_sections();

  if( default_microenvironment_options.simulate_2D == true ){
        int n = 0;
        while( y < tumor_radius )
        {
            x = 0.0;
            if( n % 2 == 1 )
            { x = 0.5*cell_spacing; }
            x_outer = sqrt( tumor_radius*tumor_radius - y*y );

            while( x < x_outer )
            {
                pCell = create_cell(); // tumor cell
                pCell->assign_position( x , y , 0.0 );


                if( fabs( y ) > 0.01 )
                {
                    pCell = create_cell(); // tumor cell
                    pCell->assign_position( x , -y , 0.0 );
                }

                if( fabs( x ) > 0.01 )
                {
                    pCell = create_cell(); // tumor cell
                    pCell->assign_position( -x , y , 0.0 );

                    if( fabs( y ) > 0.01 )
                    {
                        pCell = create_cell(); // tumor cell
                        pCell->assign_position( -x , -y , 0.0 );

                    }
                }
                x += cell_spacing;

            }

            y += cell_spacing * sqrt(3.0)/2.0;
            n++;
        }
	} else {
        std::vector<std::vector<double>> positions = create_cell_sphere_positions(cell_radius,tumor_radius);
        for( int i=0; i < positions.size(); i++ ){
            pCell = create_cell(); // tumor cell
            pCell->assign_position( positions[i] );
        }
    }

	return;
}

// custom cell phenotype function
void tumor_cell_phenotype( Cell* pCell, Phenotype& phenotype, double dt )
{
	static int genes_i = 0;
	static int proteins_i =1;
	static int creation_rates_i = 2;
	static int degradation_rates_i = 3;

	static int red_i = 0;
	static int green_i = 1;

	static int persistence_time_i = pCell->custom_data.find_variable_index( "persistence time" );
	static int necrosis_index = cell_defaults.phenotype.death.find_death_model_index( PhysiCell_constants::necrosis_death_model );
	static int oxygen_i = get_default_microenvironment()->find_density_index( "oxygen" );


    // update proliferation rate
	double pO2 = (pCell->nearest_density_vector())[oxygen_i];
    double multiplier = 1.0;
	if( pO2 < pCell->parameters.o2_proliferation_threshold)
    {
        multiplier = 0.0;
    }
	else{
		if( pO2 < pCell->parameters.o2_proliferation_saturation )
		{
			multiplier = ( pO2 - pCell->parameters.o2_proliferation_threshold ) / ( pCell->parameters.o2_proliferation_saturation - pCell->parameters.o2_proliferation_threshold );
		}
	}
	phenotype.cycle.data.transition_rate(0,1) = multiplier * cell_defaults.phenotype.cycle.data.transition_rate(0,1);

	// deterministic necrosis
	if( pO2 < pCell->parameters.o2_necrosis_threshold )
	{
		phenotype.death.rates[necrosis_index] = 9e99;
	}

    // if cell is dead, don't bother with future phenotype changes.
	if( phenotype.death.dead == true )
	{
		pCell->functions.update_phenotype = NULL;
		return;
	}

	// set hypoxia threshold
	static double FP_hypoxic_switch = parameters.doubles["sigma_H"].value;

	// permanent gene switch
	if( pO2 < FP_hypoxic_switch )
	{
		pCell->custom_data.vector_variables[genes_i].value[red_i] = 0.0;
		pCell->custom_data.vector_variables[genes_i].value[green_i] = 1.0;
	}

	// update the proteins
	for( int i=0; i < pCell->custom_data.vector_variables[genes_i].value.size(); i++ )
	{
		double temp = pCell->custom_data.vector_variables[creation_rates_i].value[i]; // alpha_i
		temp += pCell->custom_data.vector_variables[degradation_rates_i].value[i]; // alpha_i + beta_i
		temp *= pCell->custom_data.vector_variables[genes_i].value[i]; // G_i^n ( alpha_i + beta_i );
		temp *= dt; // dt*G_i^n ( alpha_i + beta_i );
		pCell->custom_data.vector_variables[proteins_i].value[i] += temp; // P_i = P_i + dt*G_i^n ( alpha_i + beta_i );
		temp = pCell->custom_data.vector_variables[creation_rates_i].value[i]; // alpha_i
		temp *= pCell->custom_data.vector_variables[genes_i].value[i]; // G_i^n * alpha_i
		temp += pCell->custom_data.vector_variables[degradation_rates_i].value[i]; // G_i^n * alpha_i + beta_i
		temp *= dt; // dt*( G_i^n * alpha_i + beta_i );
		temp += 1.0; // 1.0 + dt*( G_i^n * alpha_i + beta_i );
		pCell->custom_data.vector_variables[proteins_i].value[i] /= temp; // P_i = ( P_i + dt*G_i^n ( alpha_i + beta_i ) ) / ( 1.0 + dt*( G_i^n * alpha_i + beta_i ) );
	}

    // change phenotype
	if( pO2 < FP_hypoxic_switch)
	{
        phenotype.motility.is_motile = true;
        phenotype.motility.migration_speed = parameters.doubles["speed_green"].value;
        phenotype.motility.persistence_time = parameters.doubles["pers_time_green"].value;
        // fraction of green cells
        int countGreenCells = 0; int countGreenCellsM = 0;
        for(int i=0;i<all_cells->size();i++){
            if((*all_cells)[i]->custom_data.vector_variables[genes_i].value[green_i] == 1.0 && (*all_cells)[i]->phenotype.cycle.current_phase().code < 100){
                countGreenCells++;
                if(parameters.doubles["bias_green_rsp"].value - (*all_cells)[i]->phenotype.motility.migration_bias < 0.001 ) countGreenCellsM++;
            }
        }
        double fractionGreenCells = countGreenCellsM/(1.0*countGreenCells);
        // choose of the bias
        if(fractionGreenCells <= parameters.doubles["fraction_rsp"].value && parameters.doubles["fraction_rsp"].value != 0)
            phenotype.motility.migration_bias = parameters.doubles["bias_green_rsp"].value;
        else{
            phenotype.motility.migration_bias = parameters.doubles["bias_green"].value;
        }
	}
	else
	{
        // just GFP+ cells have a persistence time
		if (phenotype.motility.is_motile == true && pCell->custom_data.vector_variables[genes_i].value[green_i] == 1.0)
		{
			  pCell->custom_data[persistence_time_i]+= dt;
			  if (pCell->custom_data[persistence_time_i] > parameters.doubles["hypoxia_pers_time"].value)
			  {
				  phenotype.motility.is_motile = false;
			  }

		}
	}

	// update dirichlet nodes
	// microenvironment.remove_dirichlet_node(microenvironment.nearest_voxel_index( pCell->position));
	return;
}

std::vector<std::string> AMIGOS_coloring_function( Cell* pCell )
{
	static int section_type = get_cell_definition( "blood vessel section" ).type;

	static int genes_i = 0;
	static int proteins_i =1;
	static int creation_rates_i = 2;
	static int degradation_rates_i = 3;

	static int red_i = 0;
	static int green_i = 1;

    std::vector< std::string > output( 4, "black" );

	// oxygen;
  static int oxygen_i = get_default_microenvironment()->find_density_index( "oxygen" );
	double pO2 = (pCell->nearest_density_vector())[oxygen_i];

	static int cyto_color_i = 4;
	static int nuclear_color_i = 5;

	if ( pCell->type == section_type )
	{
		output[0] = "rgb(255,0,0)";
		output[1] = "rgb(255,0,0)";
		output[2] = "rgb(255,0,0)";

		pCell->custom_data.vector_variables[cyto_color_i].value[0] = 255;
		pCell->custom_data.vector_variables[cyto_color_i].value[1] = 0;
		pCell->custom_data.vector_variables[cyto_color_i].value[2] = 0;

		pCell->custom_data.vector_variables[nuclear_color_i].value[0] = 255;
		pCell->custom_data.vector_variables[nuclear_color_i].value[1] = 0;
		pCell->custom_data.vector_variables[nuclear_color_i].value[2] = 0;

		return output;
	}

	// live cells are a combination of red and green
	if( pCell->phenotype.death.dead == false )
	{
		int red   = (int) round( pCell->custom_data.vector_variables[proteins_i].value[red_i] * 255.0 );
		int green = (int) round( pCell->custom_data.vector_variables[proteins_i].value[green_i] * 255.0);

		char szTempString [128];
        if (pO2 > parameters.doubles["sigma_H"].value || parameters.bools["hypoxyprobe"].value == false){
            sprintf( szTempString , "rgb(%u,%u,0)", red, green );
            output[0].assign( szTempString );
            output[1].assign( szTempString );

            sprintf( szTempString , "rgb(%u,%u,%u)", (int)round(output[0][0]/2.0) , (int)round(output[0][1]/2.0) , (int)round(output[0][2]/2.0) );
            output[2].assign( szTempString );

            pCell->custom_data.vector_variables[cyto_color_i].value[0] = red;
            pCell->custom_data.vector_variables[cyto_color_i].value[1] = green;
            pCell->custom_data.vector_variables[cyto_color_i].value[2] = 0.0;

            pCell->custom_data.vector_variables[nuclear_color_i].value[0] = red / 2.0;
            pCell->custom_data.vector_variables[nuclear_color_i].value[1] = green / 2.0;
            pCell->custom_data.vector_variables[nuclear_color_i].value[2] = 0.0 / 2.0;
        }
        else{ // mark hypoxyprobe
            output[0] = "rgb(128,0,128)";
		    output[2] = "rgb(64,0,64)";
            pCell->custom_data.vector_variables[cyto_color_i].value[0] = 128.0;
		    pCell->custom_data.vector_variables[cyto_color_i].value[1] = 0.0;
            pCell->custom_data.vector_variables[cyto_color_i].value[2] = 128.0;
            pCell->custom_data.vector_variables[nuclear_color_i].value[0] = 64.0;
		    pCell->custom_data.vector_variables[nuclear_color_i].value[1] = 0.0;
            pCell->custom_data.vector_variables[nuclear_color_i].value[2] = 64.0;
        }

		return output;
	}

	// if not, dead colors
	if (pCell->phenotype.cycle.current_phase().code == PhysiCell_constants::apoptotic )  // Apoptotic - Red
	{
		output[0] = "rgb(255,0,0)";
		output[2] = "rgb(125,0,0)";

		pCell->custom_data.vector_variables[cyto_color_i].value[0] = 255;
		pCell->custom_data.vector_variables[cyto_color_i].value[1] = 0;
		pCell->custom_data.vector_variables[cyto_color_i].value[2] = 0.0;

		pCell->custom_data.vector_variables[nuclear_color_i].value[0] = 125;
		pCell->custom_data.vector_variables[nuclear_color_i].value[1] = 0;
		pCell->custom_data.vector_variables[nuclear_color_i].value[2] = 0;

	}
	if( pCell->phenotype.cycle.current_phase().code == PhysiCell_constants::necrotic_swelling ||
		pCell->phenotype.cycle.current_phase().code == PhysiCell_constants::necrotic_lysed ||
		pCell->phenotype.cycle.current_phase().code == PhysiCell_constants::necrotic ) // Necrotic - Blue
	{
		output[0] = "rgb(99,54,222)";
		output[2] = "rgb(32,13,107)";

		pCell->custom_data.vector_variables[cyto_color_i].value[0] = 99;
		pCell->custom_data.vector_variables[cyto_color_i].value[1] = 54;
		pCell->custom_data.vector_variables[cyto_color_i].value[2] = 222;

		pCell->custom_data.vector_variables[nuclear_color_i].value[0] = 32;
		pCell->custom_data.vector_variables[nuclear_color_i].value[1] = 13;
		pCell->custom_data.vector_variables[nuclear_color_i].value[2] = 107;
	}

	return output;
}

void oxygen_taxis_motility( Cell* pCell, Phenotype& phenotype, double dt )
{
	static int oxygen_i = pCell->get_microenvironment()->find_density_index( "oxygen" );

	phenotype.motility.migration_bias_direction = pCell->nearest_gradient( oxygen_i );
	normalize( &(phenotype.motility.migration_bias_direction) ) ;

	return;
}
