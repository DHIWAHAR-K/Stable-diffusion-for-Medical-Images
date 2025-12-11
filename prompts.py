import re
import os

class MedicalPromptGenerator:
    def __init__(self):
        # Mappings
        self.density_map_text = {
            'A': 'fatty',
            'B': 'scattered fibroglandular',
            'C': 'heterogeneously dense',
            'D': 'extremely dense'
        }
        self.density_map_class = {
            'A': 'low density',
            'B': 'low density',
            'C': 'high density', # As requested
            'D': 'high density'  # As requested
        }
        self.view_map = {
            'CC': 'CC view',
            'MLO': 'MLO view'
        }
        self.lat_map = {
            'L': 'Left',
            'R': 'Right'
        }

    def parse_filename_attributes(self, filename):
        """
        Extracts attributes from filenames formatted like:
        814_R_MLO_BI-RADS_2_DENSITY_C_...
        """
        attrs = {}
        
        # 1. Laterality (_L_ or _R_)
        if '_L_' in filename:
            attrs['laterality'] = 'Left'
        elif '_R_' in filename:
            attrs['laterality'] = 'Right'
        else:
            attrs['laterality'] = None

        # 2. View (_CC_ or _MLO_)
        if '_CC_' in filename:
            attrs['view'] = 'CC'
        elif '_MLO_' in filename:
            attrs['view'] = 'MLO'
        else:
            attrs['view'] = None

        # 3. BIRADS (BI-RADS_X)
        birads_match = re.search(r'BI-RADS_(\d+)', filename)
        if birads_match:
            attrs['birads'] = birads_match.group(1)
        else:
            attrs['birads'] = None

        # 4. Density (DENSITY_X)
        density_match = re.search(r'DENSITY_([ABCD])', filename)
        if density_match:
            attrs['density_grade'] = density_match.group(1)
        else:
            attrs['density_grade'] = None
            
        return attrs

    def generate_prompt(self, filename, fallback_density_class=None):
        """
        Constructs a detailed clinical prompt.
        """
        attrs = self.parse_filename_attributes(filename)
        
        # Components
        view_str = self.view_map.get(attrs['view'], "screening")
        lat_str = attrs['laterality'] if attrs['laterality'] else "breast"
        
        # Density Logic
        density_grade = attrs['density_grade']
        if not density_grade and fallback_density_class:
            # Try to infer from class label "DENSITY A" etc if passed
            if "DENSITY" in fallback_density_class:
                density_grade = fallback_density_class.split(' ')[-1] # "DENSITY A" -> "A"
        
        if density_grade:
            density_desc = self.density_map_class.get(density_grade, "medical")
            density_text = f"{density_desc} tissue (Density {density_grade})"
        else:
            density_text = "mammogram tissue"

        # BIRADS Logic
        birads_score = attrs['birads']
        if birads_score:
            birads_text = f" with BIRADS {birads_score} findings"
        else:
            birads_text = ""

        # Construct Sentence
        # Template: "A {View} mammogram of the {Lat} breast showing {Density} {BIRADS}."
        
        if attrs['laterality']:
             prompt = f"A {view_str} mammogram of the {lat_str} breast showing {density_text}{birads_text}."
        else:
             prompt = f"A {view_str} mammogram showing {density_text}{birads_text}."

        return prompt

if __name__ == "__main__":
    # Test
    gen = MedicalPromptGenerator()
    fname = "814_R_MLO_BI-RADS_2_DENSITY_C_hash.png"
    print(fname)
    print(gen.generate_prompt(fname))
    
    fname2 = "001_L_CC_BI-RADS_1_DENSITY_A_hash.png"
    print(fname2)
    print(gen.generate_prompt(fname2))
