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

#include "./proliferation2D.h"


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
	
	cell_defaults.functions.set_orientation = up_orientation; 
	cell_defaults.phenotype.geometry.polarity = 1.0; 
	cell_defaults.phenotype.motility.restrict_to_2D = true; 
	
	// use default proliferation and death 
	
	/* int cycle_start_index = live.find_phase_index( PhysiCell_constants::live ); 
	int cycle_end_index = live.find_phase_index( PhysiCell_constants::live );  */

	int apoptosis_index = cell_defaults.phenotype.death.find_death_model_index( PhysiCell_constants::apoptosis_death_model );
	
	cell_defaults.parameters.o2_proliferation_saturation = parameters.doubles["sigmaS"].value;   
	cell_defaults.parameters.o2_reference = cell_defaults.parameters.o2_proliferation_saturation;
	cell_defaults.parameters.o2_proliferation_threshold = parameters.doubles["sigmaT"].value;

	cell_defaults.phenotype.cycle.data.transition_rate(0,1) = parameters.doubles["rate_KnToKp"].value;
	cell_defaults.phenotype.cycle.data.transition_rate(1,0) = parameters.doubles["rate_KpToKn"].value;
	
	// set default motiltiy
	cell_defaults.phenotype.motility.is_motile = false; 
	cell_defaults.functions.update_migration_bias = oxygen_taxis_motility;
	
	// set default uptake and secretion 
	// oxygen 
	cell_defaults.phenotype.secretion.secretion_rates[oxygen_i] = 0; 
	cell_defaults.phenotype.secretion.uptake_rates[oxygen_i] = parameters.doubles["uptake_rate"].value;
	
	// set the default cell type to no phenotype updates 
	
	cell_defaults.functions.update_phenotype = tumor_cell_phenotype;
	
	cell_defaults.name = "cancer cell"; 
	cell_defaults.type = 0;
	
	return; 
}

void setup_microenvironment( void )
{
	// make sure ot override and go back to 2D 
	if( default_microenvironment_options.simulate_2D == false )
	{
		std::cout << "Warning: overriding XML config option and setting to 2D!" << std::endl; 
		default_microenvironment_options.simulate_2D = true; 
	}	
	initialize_microenvironment();
	
	std::vector< double > position = {0.0,0.0,0.0};
	for( unsigned int n=0; n < microenvironment.number_of_voxels() ; n++ ){
		if (dist(microenvironment.mesh.voxels[n].center,position) >= parameters.doubles["tumor_radius"].value)
			microenvironment.add_dirichlet_node(n,default_microenvironment_options.Dirichlet_condition_vector);
	}

	return;
}	

void setup_tissue( void )
{		
	// place a cluster of tumor cells at the center 
	
	/* double cell_radius = cell_defaults.phenotype.geometry.radius; 
	double cell_spacing = 0.95 * 2.0 * cell_radius;  */
	
	double tumor_radius = parameters.doubles["tumor_radius"].value; 
		
	Cell* pCell = NULL; 
	
	double x = 0.0; 
	double x_outer = tumor_radius; 
	double y = 0.0; 
	double z = 0.0;
	
	int n = 0;
	
	std::ifstream readFile("Posfile.txt"); 
	while (!readFile.eof()){
		readFile >> x >> y >> z;
		if (sqrt(x*x+y*y+z*z) < tumor_radius){
			pCell = create_cell();
			pCell->assign_position( x , y , z);
		}
	}
	readFile.close();
	
	/* int n = 0; 
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
	} */
		
	return; 
}

// custom cell phenotype function to scale immunostimulatory factor with hypoxia 
void tumor_cell_phenotype( Cell* pCell, Phenotype& phenotype, double dt )
{
	static int necrosis_index = cell_defaults.phenotype.death.find_death_model_index( PhysiCell_constants::necrosis_death_model );
	static int oxygen_i = get_default_microenvironment()->find_density_index( "oxygen" );
	//update_cell_and_death_parameters_O2_based(pCell,phenotype,dt);
	double pO2 = (pCell->nearest_density_vector())[oxygen_i];
    double multiplier = 1.0;
	if( pO2 < pCell->parameters.o2_proliferation_threshold || pCell->state.simple_pressure > 5.0)
    { 
        multiplier = 0.0;
    }
	else{
		if( pO2 < pCell->parameters.o2_proliferation_saturation )
		{
			multiplier = ( pO2 - pCell->parameters.o2_proliferation_threshold ) / ( pCell->parameters.o2_proliferation_saturation - pCell->parameters.o2_proliferation_threshold );
			multiplier = pow(multiplier,parameters.doubles["expoent_o2_update"].value);//0.32;
		}
	}	
	phenotype.cycle.data.transition_rate(0,1) = multiplier * cell_defaults.phenotype.cycle.data.transition_rate(0,1);
	
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
	
	//Update dirichlet nodes
	microenvironment.remove_dirichlet_node(microenvironment.nearest_voxel_index( pCell->position));
	
	return; 
}

std::vector<std::string> AMIGOS_coloring_function( Cell* pCell )
{
	std::vector< std::string > output( 4, "black" ); 
	// live cells are a combination of red and green 
	if( pCell->phenotype.death.dead == false )
	{
		int red   = 255.0; 
		int green = 0; 
		
		char szTempString [128];
		sprintf( szTempString , "rgb(%u,%u,0)", red, green );
		output[0].assign( szTempString );
		output[1].assign( szTempString );

		sprintf( szTempString , "rgb(%u,%u,%u)", (int)round(output[0][0]/2.0) , (int)round(output[0][1]/2.0) , (int)round(output[0][2]/2.0) );
		output[2].assign( szTempString );
		
		return output; 
	}
	
	// Necrotic - Brown
	if( pCell->phenotype.cycle.current_phase().code == PhysiCell_constants::necrotic_swelling || 
		pCell->phenotype.cycle.current_phase().code == PhysiCell_constants::necrotic_lysed || 
		pCell->phenotype.cycle.current_phase().code == PhysiCell_constants::necrotic )
	{
		output[0] = "rgb(250,138,38)";
		output[2] = "rgb(139,69,19)";
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

void QOI(const char* FILE)
{
	std::ofstream OutFile (FILE);
	std::vector<double> KI67pos(60);
	std::vector<double> KI67neg(60);
	
	std::vector< double > position = {0.0,0.0,0.0};
	
	for(int i=0;i<all_cells->size();i++){
		double distance = dist(position,(*all_cells)[i]->position);
		int index = distance/50;
		//std::cout << "Phase: " << (*all_cells)[i]->phenotype.cycle.data.current_phase_index << std::endl;
		//std::cout << " Rates: "<< (*all_cells)[i]->phenotype.cycle.data.transition_rate(1,0) << " " << (*all_cells)[i]->phenotype.cycle.data.transition_rate(0,1) << std::endl;
		if((*all_cells)[i]->phenotype.cycle.data.current_phase_index == 1){
			KI67pos[index] += 1; 
		}
		if((*all_cells)[i]->phenotype.cycle.data.current_phase_index == 0){
			KI67neg[index] += 1;
		}
	}
	double temp;
	for(int j=0;j < KI67pos.size();j++){
		temp = KI67pos[j];
		KI67pos[j] = temp/(temp+KI67neg[j]);
		KI67neg[j] = KI67neg[j]/(temp+KI67neg[j]);
		if ((temp+KI67neg[j]) == 0){KI67pos[j] = 0; KI67neg[j] = 0;}  
		OutFile << std::scientific <<   j*0.05 + 0.025 << "\t" << KI67pos[j] << "\t" << KI67neg[j] <<std::endl;
	}
	OutFile.close(); 

/* 	for(int j=0;j < KI67.size();j++){
		position[1] += 100;
		std::vector< double > Value = microenvironment.nearest_density_vector( position );
		double pO2 = Value[0]; 
		double multiplier = 1.0;
		if( pO2 < parameters.doubles["prol_saturation"].value )
		{
			multiplier = ( pO2 - parameters.doubles["prol_threshold"].value ) 
				 / ( parameters.doubles["prol_saturation"].value - parameters.doubles["prol_threshold"].value );
		}
		if( pO2 < parameters.doubles["prol_threshold"].value )
		{ 
			multiplier = 0.0; 
		}
		 
		OutFile <<  (j+1)*0.1 << "\t" << multiplier <<  "\t" << pO2 << std::endl;
	} */
	return; 
}
