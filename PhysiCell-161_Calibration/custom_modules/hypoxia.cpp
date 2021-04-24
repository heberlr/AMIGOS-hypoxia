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


void create_cell_types( void )
{
	// use the same random seed so that future experiments have the 
	// same initial histogram of oncoprotein, even if threading means 
	// that future division and other events are still not identical 
	// for all runs 
	SeedRandom(0); 
	
	int oxygen_i = get_default_microenvironment()->find_density_index( "oxygen" );
	
	// housekeeping 
	
	initialize_default_cell_definition();
	cell_defaults.phenotype.secretion.sync_to_microenvironment( &microenvironment ); 
	
	// turn the default cycle model to live, 
	// so it's easier to turn off proliferation
	
	cell_defaults.phenotype.cycle.sync_to_cycle_model( Ki67_basic ); 
	
	// Make sure we're ready for 2D
    if( default_microenvironment_options.simulate_2D == true ){
        cell_defaults.functions.set_orientation = up_orientation; 
        cell_defaults.phenotype.geometry.polarity = 1.0; 
        cell_defaults.phenotype.motility.restrict_to_2D = true;
    }
	
	// use default proliferation and death 
	
	/* int cycle_start_index = live.find_phase_index( PhysiCell_constants::live ); 
	int cycle_end_index = live.find_phase_index( PhysiCell_constants::live );  */
	
	int apoptosis_index = cell_defaults.phenotype.death.find_death_model_index( PhysiCell_constants::apoptosis_death_model );
	
	cell_defaults.parameters.o2_proliferation_saturation = parameters.doubles["sigmaS"].value;//38;   
	cell_defaults.parameters.o2_reference = cell_defaults.parameters.o2_proliferation_saturation;
	cell_defaults.parameters.o2_proliferation_threshold = parameters.doubles["sigmaT"].value;//1.6;
	cell_defaults.parameters.o2_necrosis_threshold = cell_defaults.parameters.o2_proliferation_threshold;
	cell_defaults.parameters.o2_necrosis_max = cell_defaults.parameters.o2_proliferation_threshold;
	
	cell_defaults.phenotype.cycle.data.transition_rate(0,1) = parameters.doubles["rate_KnToKp"].value;//0.0010;
	cell_defaults.phenotype.cycle.data.transition_rate(1,0) = parameters.doubles["rate_KpToKn"].value;//0.0017;
	
	// set default motiltiy
	cell_defaults.phenotype.motility.is_motile = true; 
	cell_defaults.functions.update_migration_bias = oxygen_taxis_motility; 
			
	cell_defaults.phenotype.motility.persistence_time = parameters.doubles["perst_time_red"].value;//20.35; 
	cell_defaults.phenotype.motility.migration_bias = parameters.doubles["bias_red"].value;//0.1645; 
	cell_defaults.phenotype.motility.migration_speed = parameters.doubles["speed_red"].value;//0.2769; 
	
	// set default uptake and secretion 
	// oxygen 
	cell_defaults.phenotype.secretion.secretion_rates[oxygen_i] = 0; 
	cell_defaults.phenotype.secretion.uptake_rates[oxygen_i] = parameters.doubles["uptake_rate"].value; //0.4663; 

	// set the default cell type to no phenotype updates 
	
	cell_defaults.functions.update_phenotype = tumor_cell_phenotype; 
	
	cell_defaults.name = "cancer cell"; 
	cell_defaults.type = 0; 
	
	// add custom data 
	
	std::vector<double> genes = { 1.0, 0.0 }; // RFP, GFP 
	std::vector<double> proteins = {1.0, 0.0 }; // RFP, GFP; 
	
	double default_degradation_rate = 6.8e-5; // 7-day half-life 
	// 4.8e-4; // 24 hour half-life 
	// 0.0077; // 90 minute half-life 
	// 0.019; // 90% degrades in 120 minutes 
	
	std::vector<double> degradation_rates = { default_degradation_rate , default_degradation_rate }; 
	
	double default_production_rate = 4.8e-4; // 24 hour half ramp-up 
	// 0.0019; // 6 hours to reach 50% 
	// 0.0068; // 1.7 hours to reach 50% 
	// 0.23; // 10 minutes to reach 90% 
	
	std::vector<double> creation_rates = { default_production_rate , default_production_rate }; 
	
	cell_defaults.custom_data.add_vector_variable( "genes" , "dimensionless", genes ); 
	
	cell_defaults.custom_data.add_vector_variable( "proteins" , "dimensionless", proteins ); 
	cell_defaults.custom_data.add_vector_variable( "creation_rates" , "1/min" , creation_rates ); 
	cell_defaults.custom_data.add_vector_variable( "degradation_rates" , "1/min" , degradation_rates ); 

	cell_defaults.custom_data.add_variable( "persistence time" , "dimensionless" , 0.0 ); 
    
    std::vector<double> color = {255, 255, 255};
	cell_defaults.custom_data.add_vector_variable( "nuclear_color" , "dimensionless", color ); 
	cell_defaults.custom_data.add_vector_variable( "cytoplasmic_color" , "dimensionless", color ); 
	
	//cell_defaults.phenotype.mechanics.cell_cell_repulsion_strength *= 5.0;
	
	return; 
}

void setup_microenvironment( void )
{
	initialize_microenvironment();
	
	std::vector< double > position = {0.0,0.0,0.0};
	for( unsigned int n=0; n < microenvironment.number_of_voxels() ; n++ ){
		if (dist(microenvironment.mesh.voxels[n].center,position) >= parameters.doubles["tumor_radius"].value)
			microenvironment.add_dirichlet_node(n,default_microenvironment_options.Dirichlet_condition_vector);
	}
	
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
	
	double tumor_radius = parameters.doubles["tumor_radius"].value; 
		
	Cell* pCell = NULL; 
	
	double x = 0.0; 
	double x_outer = tumor_radius; 
	double y = 0.0; 

	
	
	
	//std::ifstream readFile("Posfile.txt"); 
    /*double z = 0.0;
    std::ifstream readFile(parameters.strings["FileInitialCond"].value);
    if (readFile.is_open()){
        while (!readFile.eof()){
            readFile >> x >> y >> z;
            if (sqrt(x*x+y*y+z*z) < tumor_radius){
                pCell = create_cell();
                pCell->assign_position( x , y , z);
            }
        }
    }
    else exit(-1);
	readFile.close();*/
	
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

// custom cell phenotype function to scale immunostimulatory factor with hypoxia 
void tumor_cell_phenotype( Cell* pCell, Phenotype& phenotype, double dt )
{
	//std::cout << "Position: ( " << pCell->position[0] << " " << pCell->position[1] <<" " << pCell->position[2] << ") -- (" << "type: " << pCell->type << std::endl;
	static int genes_i = 0; 
	static int proteins_i =1; 
	static int creation_rates_i = 2; 
	static int degradation_rates_i = 3; 
	
	static int red_i = 0; 
	static int green_i = 1; 	

	static int persistence_time_i = pCell->custom_data.find_variable_index( "persistence time" );
	static int necrosis_index = cell_defaults.phenotype.death.find_death_model_index( PhysiCell_constants::necrosis_death_model );
	static int oxygen_i = get_default_microenvironment()->find_density_index( "oxygen" ); 
	
	//update_cell_and_death_parameters_O2_based(pCell,phenotype,dt);
	double pO2 = (pCell->nearest_density_vector())[oxygen_i];
    double multiplier = 1.0;
	//if( pO2 < pCell->parameters.o2_proliferation_threshold || pCell->state.simple_pressure > 1.0)
	if( pO2 < pCell->parameters.o2_proliferation_threshold)
    { 
        multiplier = 0.0;
    }
	else{
		if( pO2 < pCell->parameters.o2_proliferation_saturation )
		{
			multiplier = ( pO2 - pCell->parameters.o2_proliferation_threshold ) / ( pCell->parameters.o2_proliferation_saturation - pCell->parameters.o2_proliferation_threshold );
			//multiplier = pow(multiplier,parameters.doubles["expoent_o2_update"].value);//0.32;
		}
	}	
	phenotype.cycle.data.transition_rate(0,1) = multiplier * cell_defaults.phenotype.cycle.data.transition_rate(0,1);
	
	//Proliferative or migrating
	/* if (pCell->phenotype.cycle.data.current_phase_index == 1) phenotype.motility.is_motile = false;
	if (pCell->phenotype.cycle.data.current_phase_index == 0) phenotype.motility.is_motile = true; */
	
	// Deterministic necrosis 
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

	// multiply proliferation rate by the oncoprotein 
	/*
	static int cycle_start_index = live.find_phase_index( PhysiCell_constants::live ); 
	static int cycle_end_index = live.find_phase_index( PhysiCell_constants::live ); 
	
	phenotype.cycle.data.transition_rate( cycle_start_index ,cycle_end_index ) *= pCell->custom_data[oncoprotein_i] ; */

	// set genes 
	
	static double FP_hypoxic_switch = 10.0;
	static double phenotype_hypoxic_switch = 10.0;  // 
	
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
	
	if( pO2 < phenotype_hypoxic_switch)
	{
        // Fraction of green cells
        int countGreenCells = 0; int countGreenCellsM = 0;
        for(int i=0;i<all_cells->size();i++){ 
            if((*all_cells)[i]->custom_data.vector_variables[genes_i].value[green_i] == 1.0 && (*all_cells)[i]->phenotype.cycle.current_phase().code < 100){ 
                countGreenCells++;
                if(parameters.doubles["bias_greenResp"].value - (*all_cells)[i]->phenotype.motility.migration_bias < 0.001 ) countGreenCellsM++;
            }
        }
        double fractionGreenCells = countGreenCellsM/(1.0*countGreenCells);
        phenotype.motility.is_motile = true; 
        phenotype.motility.migration_speed = parameters.doubles["speed_green"].value;
        phenotype.motility.persistence_time = parameters.doubles["perst_time_green"].value;
        if(fractionGreenCells <= parameters.doubles["fracResponse_green"].value && parameters.doubles["fracResponse_green"].value != 0)
            phenotype.motility.migration_bias = parameters.doubles["bias_greenResp"].value;
        else{
            phenotype.motility.migration_bias = parameters.doubles["bias_green"].value;
        }
	}
	else
	{
        // Just Green cells have a persistence time
		if (phenotype.motility.is_motile == true && pCell->custom_data.vector_variables[genes_i].value[green_i] == 1.0)
		{
            if (parameters.bools["model_transient"].value){ // Transient model
              static double probability = dt/parameters.doubles["persitence_timeHip"].value;
              if( UniformRandom() < probability )
			  {
				  phenotype.motility.is_motile = false; 
			  }
            }else{ // Deterministic model
			  pCell->custom_data[persistence_time_i]+= dt;
			  if (pCell->custom_data[persistence_time_i] > parameters.doubles["persitence_timeHip"].value)
			  {
				  phenotype.motility.is_motile = false;
			  }
            }
		}
	}
	
	//Update dirichlet nodes
	microenvironment.remove_dirichlet_node(microenvironment.nearest_voxel_index( pCell->position));
	return; 
}

std::vector<std::string> AMIGOS_coloring_function( Cell* pCell )
{
	static int genes_i = 0; 
	static int proteins_i =1; 
	static int creation_rates_i = 2; 
	static int degradation_rates_i = 3; 
	
	static int red_i = 0; 
	static int green_i = 1;
	
        std::vector< std::string > output( 4, "black" ); 
    
	//Oxygen;
        static int oxygen_i = get_default_microenvironment()->find_density_index( "oxygen" ); 
	double pO2 = (pCell->nearest_density_vector())[oxygen_i];

	static int cyto_color_i = 4;
	static int nuclear_color_i = 5;

	
	// live cells are a combination of red and green 
	if( pCell->phenotype.death.dead == false )
	{
		int red   = (int) round( pCell->custom_data.vector_variables[proteins_i].value[red_i] * 255.0 ); 
		int green = (int) round( pCell->custom_data.vector_variables[proteins_i].value[green_i] * 255.0); 

		char szTempString [128];
        if (pO2 > parameters.doubles["sigma_Hp"].value || parameters.bools["hypoxyprobe"].value == false){
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
        else{
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
	
	// Necrotic - Blue
	if( pCell->phenotype.cycle.current_phase().code == PhysiCell_constants::necrotic_swelling || 
		pCell->phenotype.cycle.current_phase().code == PhysiCell_constants::necrotic_lysed || 
		pCell->phenotype.cycle.current_phase().code == PhysiCell_constants::necrotic )
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

void QOI(double& RedVolume, double& GreenVolume, double& DoubleVolume)
{
	RedVolume = 0;
	GreenVolume = 0;
	DoubleVolume = 0;
	static int proteins_i =1;
	static double delta = 0.2;
	
	//std::cout << " Rates: "<< (*all_cells)[0]->phenotype.cycle.data.transition_rate(0,1) << " " << (*all_cells)[0]->phenotype.cycle.data.transition_rate(1,0) << std::endl;
	//std::cout << " Rates: "<< (*all_cells)[all_cells->size()-1]->phenotype.cycle.data.transition_rate(0,1) << " " << (*all_cells)[all_cells->size()-1]->phenotype.cycle.data.transition_rate(1,0) << std::endl;
	for(int i=0;i<all_cells->size();i++)
	{
		double diff = (*all_cells)[i]->custom_data.vector_variables[proteins_i].value[0] - (*all_cells)[i]->custom_data.vector_variables[proteins_i].value[1];
		if (diff > 0 && abs(diff) > delta)
		{
			RedVolume += (*all_cells)[i]->phenotype.volume.total;
		}			
		else{
			if (diff < 0 && abs(diff) > delta){ GreenVolume += (*all_cells)[i]->phenotype.volume.total;}
			else{ DoubleVolume += (*all_cells)[i]->phenotype.volume.total;}
		}
  	}
	return; 
}

void QOI2(const char* FILE)
{
	std::ofstream OutFile (FILE);
	std::vector<double> Simp_pressure(50);
	std::vector<int> Count(50);
	std::vector< double > position = {0.0,0.0,0.0};
	
	for(int i=0;i<all_cells->size();i++){
		double distance = dist(position,(*all_cells)[i]->position);
		int index = distance/60;
		//std::cout << "Phase: " << (*all_cells)[i]->phenotype.cycle.data.current_phase_index << std::endl;
		//std::cout << " Rates: "<< (*all_cells)[i]->phenotype.cycle.data.transition_rate(1,0) << " " << (*all_cells)[i]->phenotype.cycle.data.transition_rate(0,1) << std::endl;
		if((*all_cells)[i]->phenotype.cycle.data.current_phase_index == 1 || (*all_cells)[i]->phenotype.cycle.data.current_phase_index == 0){
			Simp_pressure[index] += (*all_cells)[i]->state.simple_pressure;
			Count[index] += 1;
		}
	}
	for(int j=0;j < Simp_pressure.size();j++){
		Simp_pressure[j] = Simp_pressure[j]/Count[j];
		if (Count[j] == 0) Simp_pressure[j] = 0.0;
		OutFile << std::scientific <<   j*50 + 25 << "\t" << Simp_pressure[j] << "\t" << Count[j] <<std::endl;
	}
	OutFile.close();
}
