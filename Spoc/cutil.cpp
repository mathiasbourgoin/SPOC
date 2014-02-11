/******************************************************************************
 * Mathias Bourgoin, Université Pierre et Marie Curie (2011)
 *
 * Mathias.Bourgoin@gmail.com
 *
 * This software is a computer program whose purpose is to allow
 * GPU programming with the OCaml language.
 *
 * This software is governed by the CeCILL-B license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL-B
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and,  more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-B license and that you accept its terms.
*******************************************************************************/
#include "cutil.h"

char*
cutFindFilePath(const char* filename, const char* executable_path)
{
    // search in data/
    if (filename == 0)
        return 0;
    size_t filename_len = strlen(filename);
    const char data_folder[] = "data/";
    size_t data_folder_len = strlen(data_folder);
    char* file_path =
      (char*) malloc( sizeof(char) * (data_folder_len + filename_len + 1));
    strcpy(file_path, data_folder);
    strcat(file_path, filename);
	size_t file_path_len = strlen(file_path);
	file_path[file_path_len] = '\0';
    std::fstream fh0(file_path, std::fstream::in);
    if (fh0.good())
        return file_path;
    free( file_path);

    // search in ../../../src/<executable_name>/data/
    if (executable_path == 0)
        return 0;
    size_t executable_path_len = strlen(executable_path);
    const char* exe;
    for (exe = executable_path + executable_path_len - 1;
         exe >= executable_path; --exe)
        if (*exe == '/' || *exe == '\\')
            break;
    if (exe < executable_path)
        exe = executable_path;
    else
        ++exe;
    size_t executable_len = strlen(exe);
    size_t executable_dir_len = executable_path_len - executable_len;
    const char projects_relative_path[] = "../../../src/";
    size_t projects_relative_path_len = strlen(projects_relative_path);
    file_path =
      (char*) malloc( sizeof(char) * (executable_path_len +
         projects_relative_path_len + 1 + data_folder_len + filename_len + 1));
    strncpy(file_path, executable_path, executable_dir_len);
    file_path[executable_dir_len] = '\0';
    strcat(file_path, projects_relative_path);
    strcat(file_path, exe);
    file_path_len = strlen(file_path);
    if (*(file_path + file_path_len - 1) == 'e' &&
        *(file_path + file_path_len - 2) == 'x' &&
        *(file_path + file_path_len - 3) == 'e' &&
        *(file_path + file_path_len - 4) == '.') {
        *(file_path + file_path_len - 4) = '/';
        *(file_path + file_path_len - 3) = '\0';
    }
    else {
        *(file_path + file_path_len - 0) = '/';
        *(file_path + file_path_len + 1) = '\0';
    }
    strcat(file_path, data_folder);
    strcat(file_path, filename);
	file_path_len = strlen(file_path);
	file_path[file_path_len] = '\0';
	std::fstream fh1(file_path, std::fstream::in);
    if (fh1.good())
        return file_path;
    free( file_path);
    return 0;
}

bool inline
findModulePath(const char *module_file, string & module_path, char **argv, string & ptx_source)
{
    module_path = cutFindFilePath(module_file, argv[0]);
    if (module_path.empty()) {
       printf("> findModulePath could not find file: <%s> \n", module_file);
       return false;
    } else {
       printf("> findModulePath found file at <%s>\n", module_path.c_str());

	   if (module_path.rfind(".ptx") != string::npos) {
		   FILE *fp = fopen(module_path.c_str(), "rb");
		   fseek(fp, 0, SEEK_END);
		   int file_size = ftell(fp);
           char *buf = new char[file_size+1];
           fseek(fp, 0, SEEK_SET);
           fread(buf, sizeof(char), file_size, fp);
           fclose(fp);
           buf[file_size] = '\0';
           ptx_source = buf;
           delete[] buf;
	   }
	   return true;
    }
}
