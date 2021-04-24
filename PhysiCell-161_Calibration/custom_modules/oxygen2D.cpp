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

#include "./oxygen2D.h"


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
	
	
	// Make sure we're ready for 2D
	
	cell_defaults.functions.set_orientation = up_orientation; 
	cell_defaults.phenotype.geometry.polarity = 1.0; 
	cell_defaults.phenotype.motility.restrict_to_2D = true; 
	
	/// turn off proliferation and apoptosis; 
	int cycle_start_index = live.find_phase_index( PhysiCell_constants::live ); 
	int cycle_end_index = live.find_phase_index( PhysiCell_constants::live ); 
	cell_defaults.phenotype.cycle.data.transition_rate(cycle_start_index,cycle_end_index) = 0.0; 
	int apoptosis_index = cell_defaults.phenotype.death.find_death_model_index( PhysiCell_constants::apoptosis_death_model );
	cell_defaults.phenotype.death.rates[apoptosis_index] = 0;
	
	// set default motiltiy
	cell_defaults.phenotype.motility.is_motile = false;  
	
	// set default uptake and secretion 
	// oxygen 
	cell_defaults.phenotype.secretion.secretion_rates[oxygen_i] = 0; 
	cell_defaults.phenotype.secretion.uptake_rates[oxygen_i] = parameters.doubles["uptake_rate"].value; 

	// set the default cell type to no phenotype updates 
	
	cell_defaults.functions.update_phenotype = tumor_cell_phenotype;
	
	
	cell_defaults.phenotype.mechanics.cell_cell_adhesion_strength = 0.0;
	cell_defaults.phenotype.mechanics.cell_BM_adhesion_strength = 0.0;
	cell_defaults.phenotype.mechanics.cell_cell_repulsion_strength = 0.0;
	cell_defaults.phenotype.mechanics.cell_BM_repulsion_strength = 0.0;
	
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
		if (dist(microenvironment.mesh.voxels[n].center,position) >= parameters.doubles( "tumor_radius" ))
			microenvironment.add_dirichlet_node(n,default_microenvironment_options.Dirichlet_condition_vector);
	}

	return; 
}	

void setup_tissue( void )
{		
	// place a cluster of tumor cells at the center 
	
	double cell_radius = cell_defaults.phenotype.geometry.radius; 
	double cell_spacing = 0.95 * 2.0 * cell_radius; 
	
	double tumor_radius = tumor_radius = parameters.doubles( "tumor_radius" ); 
		
	Cell* pCell = NULL; 
	
	double x = 0.0; 
	double x_outer = tumor_radius; 
	double y = 0.0; 
	double z = 0.0;
	
	int n = 0; 
	
	std::ifstream readFile("Posfile.txt"); 
	while (!readFile.eof()){
		readFile >> x >> y >> z;
		pCell = create_cell();
		pCell->assign_position( x , y , z);
		if (sqrt(x*x+y*y+z*z) < 570.0)
			pCell->phenotype.secretion.uptake_rates[0] = 0.1*parameters.doubles["uptake_rate"].value;
	}
	readFile.close();
	
	/* while( y < tumor_radius )
	{
		x = 0.0; 
		if( n % 2 == 1 )
		{ x = 0.5*cell_spacing; }
		x_outer = sqrt( tumor_radius*tumor_radius - y*y ); 
		
		while( x < x_outer )
		{
			pCell = create_cell(); // tumor cell 
			pCell->assign_position( x , y , 0.0 );
			if (sqrt(x*x+y*y) < 570.0)
				pCell->phenotype.secretion.uptake_rates[0] = 0.1*parameters.doubles["uptake_rate"].value;
					
			
			if( fabs( y ) > 0.01 )
			{
				pCell = create_cell(); // tumor cell 
				pCell->assign_position( x , -y , 0.0 );
				if (sqrt(x*x+y*y) < 570.0)
					pCell->phenotype.secretion.uptake_rates[0] = 0.1*parameters.doubles["uptake_rate"].value;
			}
			
			if( fabs( x ) > 0.01 )
			{ 
				pCell = create_cell(); // tumor cell 
				pCell->assign_position( -x , y , 0.0 );
				if (sqrt(x*x+y*y) < 570.0)
					pCell->phenotype.secretion.uptake_rates[0] = 0.1*parameters.doubles["uptake_rate"].value;
								
				if( fabs( y ) > 0.01 )
				{
					pCell = create_cell(); // tumor cell 
					pCell->assign_position( -x , -y , 0.0 );
					if (sqrt(x*x+y*y) < 570.0)
						pCell->phenotype.secretion.uptake_rates[0] = 0.1*parameters.doubles["uptake_rate"].value;
					
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
	//std::vector< double > position = {0.0,0.0,0.0};
	//double distance = dist(position,(pCell->position));
	//if (pCell->ID == 50 || pCell->ID == 20000) std::cout << "ID: " << pCell->ID << " Phase: " << pCell->phenotype.cycle.current_phase().code << " Uptake: " <<  pCell->phenotype.secretion.uptake_rates[0] << std::endl;
	//Update dirichlet nodes
	microenvironment.remove_dirichlet_node(microenvironment.nearest_voxel_index( pCell->position));
	return; 
}

std::vector<std::string> AMIGOS_coloring_function( Cell* pCell )
{
	int red   = 255.0; 
	int green = 0; 
	std::vector< std::string > output( 4, "black" ); 
	char szTempString [128];
	sprintf( szTempString , "rgb(%u,%u,0)", red, green );
	output[0].assign( szTempString );
	output[1].assign( szTempString );

	sprintf( szTempString , "rgb(%u,%u,%u)", (int)round(output[0][0]/2.0) , (int)round(output[0][1]/2.0) , (int)round(output[0][2]/2.0) );
	output[2].assign( szTempString );
	
	// Necrotic - Brown
	double x = pCell->position[0];
	double y = pCell->position[1];
	if(sqrt(x*x+y*y) < 570.0)
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
	std::vector< double > position1 = {0.0,0.0,0.0};
	std::vector< double > position2 = {0.0,500.0,0.0};
	std::vector< double > position3 = {0.0,1000.0,0.0};
	std::vector< double > position4 = {0.0,1500.0,0.0};
	std::vector< double > position5 = {0.0,2000.0,0.0};
	std::vector< double > Value1 = microenvironment.nearest_density_vector( position1 );
	std::vector< double > Value2 = microenvironment.nearest_density_vector( position2 );
	std::vector< double > Value3 = microenvironment.nearest_density_vector( position3 );
	std::vector< double > Value4 = microenvironment.nearest_density_vector( position4 );
	std::vector< double > Value5 = microenvironment.nearest_density_vector( position5 );
	std::ofstream OutFile (FILE);
	OutFile << std::scientific << 0.0 <<"\t"<< Value1[0] << std::endl;
	OutFile << std::scientific << 0.5 <<"\t"<< Value2[0] << std::endl;
	OutFile << std::scientific << 1.0 <<"\t"<< Value3[0] << std::endl;
	OutFile << std::scientific << 1.5 <<"\t"<< Value4[0] << std::endl;
	OutFile << std::scientific << 2.0 <<"\t"<< Value5[0] << std::endl;
	OutFile.close();
	return; 
}
