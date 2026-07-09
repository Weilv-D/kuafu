import xml.etree.ElementTree as ET
import os

project_file = r"c:\Users\Deng2\Desktop\temp\kuafu\stm32_firmware\MDK-ARM\F407ZG.uvprojx"

# We register the namespace to avoid generating ns0 prefixes on tags
ET.register_namespace('', '')

tree = ET.parse(project_file)
root = tree.getroot()

# Find the Groups container
groups_node = root.find('.//Groups')
if groups_node is None:
    print("Error: <Groups> tag not found in uvprojx file!")
    exit(1)

# Helper function to create a file node in Keil format
def create_file_node(file_path, file_type):
    file_node = ET.Element('File')
    
    file_name_node = ET.SubElement(file_node, 'FileName')
    file_name_node.text = os.path.basename(file_path)
    
    file_type_node = ET.SubElement(file_node, 'FileType')
    file_type_node.text = str(file_type) # 1 = C source file, 5 = Header file, 2 = ASM, etc.
    
    file_path_node = ET.SubElement(file_node, 'FilePath')
    file_path_node.text = file_path
    
    return file_node

# Helper function to add a group and its files
def add_group(groups_parent, group_name, files_list):
    # Check if group already exists
    for g in groups_parent.findall('Group'):
        name_node = g.find('GroupName')
        if name_node is not None and name_node.text == group_name:
            print(f"Group '{group_name}' already exists. Overwriting files...")
            files_container = g.find('Files')
            if files_container is not None:
                g.remove(files_container)
            files_container = ET.SubElement(g, 'Files')
            for f_path, f_type in files_list:
                files_container.append(create_file_node(f_path, f_type))
            return
            
    # Create new group
    group_node = ET.SubElement(groups_parent, 'Group')
    group_name_node = ET.SubElement(group_node, 'GroupName')
    group_name_node.text = group_name
    
    files_container = ET.SubElement(group_node, 'Files')
    for f_path, f_type in files_list:
        files_container.append(create_file_node(f_path, f_type))
    print(f"Added group '{group_name}' with {len(files_list)} files.")

# Define files list: path and type (1=C file, 5=header file)
config_files = [
    (r"..\Config\pin_config.h", 5)
]

comm_files = [
    (r"..\Comm\crc8.c", 1),
    (r"..\Comm\crc8.h", 5),
    (r"..\Comm\pi_link.c", 1),
    (r"..\Comm\pi_link.h", 5)
]

drivers_files = [
    (r"..\Drivers\bmi088.c", 1),
    (r"..\Drivers\bmi088.h", 5),
    (r"..\Drivers\ddsm315.c", 1),
    (r"..\Drivers\ddsm315.h", 5),
    (r"..\Drivers\st3215.c", 1),
    (r"..\Drivers\st3215.h", 5)
]

control_files = [
    (r"..\Control\mahony.c", 1),
    (r"..\Control\mahony.h", 5),
    (r"..\Control\kinematics.c", 1),
    (r"..\Control\kinematics.h", 5),
    (r"..\Control\lqr_controller.c", 1),
    (r"..\Control\lqr_controller.h", 5)
]

# Add groups to project
add_group(groups_node, 'Config', config_files)
add_group(groups_node, 'Comm', comm_files)
add_group(groups_node, 'Drivers_Custom', drivers_files)
add_group(groups_node, 'Control', control_files)

# Find the Application/User/Core group to add safety_state files
core_group = None
for g in groups_node.findall('Group'):
    name_node = g.find('GroupName')
    # The template group for user code is normally "Application/User/Core" or "Application/User"
    if name_node is not None and "Core" in name_node.text:
        core_group = g
        break

if core_group is None:
    # Fallback to the first group with "Application" in it
    for g in groups_node.findall('Group'):
        name_node = g.find('GroupName')
        if name_node is not None and "Application" in name_node.text:
            core_group = g
            break

if core_group is not None:
    files_container = core_group.find('Files')
    if files_container is None:
        files_container = ET.SubElement(core_group, 'Files')
    
    # Check if safety_state.c is already in there, if not add it
    has_safety_c = False
    for f in files_container.findall('File'):
        path_node = f.find('FilePath')
        if path_node is not None and "safety_state.c" in path_node.text:
            has_safety_c = True
            break
            
    if not has_safety_c:
        files_container.append(create_file_node(r"..\Core\Src\safety_state.c", 1))
        files_container.append(create_file_node(r"..\Core\Inc\safety_state.h", 5))
        print("Added safety_state files to Core group.")
else:
    print("Warning: Could not find Application/User/Core group. Please add safety_state.c manually.")

# Update Include Paths under <VariousControls>
various_controls = root.findall('.//VariousControls')
for vc in various_controls:
    include_path_node = vc.find('IncludePath')
    if include_path_node is not None:
        paths = include_path_node.text if include_path_node.text else ""
        # Append our paths if they are not already present
        custom_paths = ["..\\Config", "..\\Comm", "..\\Drivers", "..\\Control", "..\\Core\\Inc"]
        for cp in custom_paths:
            if cp not in paths:
                paths += ";" + cp
        include_path_node.text = paths
        print(f"Updated IncludePath to: {paths}")

# Write back to file
with open(project_file, 'wb') as f:
    f.write(b'<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n')
    tree.write(f, encoding='utf-8', xml_declaration=False)

print("Project configuration complete!")
