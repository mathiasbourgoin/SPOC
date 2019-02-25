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
#ifdef __cplusplus
extern "C" {
#endif


#include "Opencl_dynlink.h"

tclGetDeviceIDs* clGetDeviceIDs;
tclGetPlatformIDs* clGetPlatformIDs;
tclGetPlatformInfo* clGetPlatformInfo;
tclGetDeviceInfo* clGetDeviceInfo;
tclCreateContext* clCreateContext;
tclCreateBuffer* clCreateBuffer;
tclReleaseMemObject* clReleaseMemObject;
tclCreateCommandQueue* clCreateCommandQueue;
tclEnqueueReadBuffer* clEnqueueReadBuffer;
tclEnqueueWriteBuffer* clEnqueueWriteBuffer;
tclGetContextInfo* clGetContextInfo;
tclCreateProgramWithSource* clCreateProgramWithSource;
tclBuildProgram* clBuildProgram;
tclGetProgramBuildInfo* clGetProgramBuildInfo;
tclCreateKernel* clCreateKernel;
tclSetKernelArg* clSetKernelArg;
tclEnqueueNDRangeKernel* clEnqueueNDRangeKernel;
tclFlush* clFlush;
tclFinish* clFinish;
tclRetainCommandQueue* clRetainCommandQueue;
tclReleaseCommandQueue* clReleaseCommandQueue;
tclSetEventCallback* clSetEventCallback;


#define CL_ERROR_UNKNOWN -12

#ifdef _WIN32
    #include <Windows.h>

    #ifdef UNICODE
    static LPCWSTR __OpenCLLibName = L"opencl.dll";
    #else
    static LPCSTR __OpenCLLibName = "opencl.dll";
    #endif

    typedef HMODULE OPENCLDRIVER;

    static int LOAD_LIBRARY(OPENCLDRIVER *pInstance)
    {
        *pInstance = LoadLibrary(__OpenCLLibName);
        if (*pInstance == NULL)
        {
            printf("LoadLibrary \"%s\" failed!\n", __OpenCLLibName);
            return CL_ERROR_UNKNOWN;
        }
        return CL_SUCCESS;
    }

    #define GET_PROC_EX(name, alias, required)                              \
        alias = (t##name *)GetProcAddress(OpenCLDrvLib, #name);               \
        if (alias == NULL && required) {                                    \
            printf("Failed to find required function \"%s\" in %s\n",       \
                   #name, __OpenCLLibName);                                   \
            return CL_ERROR_UNKNOWN;                                      \
        }

#elif defined(__unix__) || defined(__APPLE__) || defined(__MACOSX)

    #include <dlfcn.h>

    #if defined(__APPLE__) || defined(__MACOSX)
    static char __OpenCLLibName[] = "/System/Library/Frameworks/OpenCL.framework/OpenCL";
    #else
  static char __OpenCLLibName[] = "libOpenCL.so";
    #endif

    typedef void * OPENCLDRIVER;

    static int  LOAD_LIBRARY(OPENCLDRIVER *pInstance)
    {
        *pInstance = dlopen(__OpenCLLibName, RTLD_NOW);
        if (*pInstance == NULL)
        {
            printf("dlopen \"%s\" failed!\n", __OpenCLLibName);
            return CL_ERROR_UNKNOWN;
        }
        return CL_SUCCESS;
    }

    #define GET_PROC_EX(name, alias, required)                              \
        alias = (t##name *)dlsym(OpenCLDrvLib, #name);                        \
        if (name == NULL && required) {                                    \
            printf("Failed to find required function \"%s\" in %s\n",       \
                    #name, __OpenCLLibName);                                  \
            return CL_ERROR_UNKNOWN;                                      \
        }

#else
#error unsupported platform
#endif

#define CHECKED_CALL(call)              \
    do {                                \
        int result = (call);       \
        if (CL_SUCCESS != result) {   \
            return result;              \
        }                               \
    } while(0)

#define GET_PROC_REQUIRED(name) GET_PROC_EX(name,name,1)
#define GET_PROC_OPTIONAL(name) GET_PROC_EX(name,name,0)
#define GET_PROC(name) GET_PROC_REQUIRED(name)

int noCL = 0;

int CL_API_ENTRY clInit(){
    	OPENCLDRIVER OpenCLDrvLib;
    	//int driverVer = 1000;

    	CHECKED_CALL(LOAD_LIBRARY(&OpenCLDrvLib));


    	GET_PROC(clGetDeviceIDs);
    	GET_PROC(clGetPlatformIDs);
    	GET_PROC(clGetPlatformInfo);
    	GET_PROC(clGetDeviceInfo);
    	GET_PROC(clCreateContext);

    	GET_PROC(clCreateBuffer);
    	GET_PROC(clReleaseMemObject);
    	GET_PROC(clCreateCommandQueue);
    	GET_PROC(clEnqueueReadBuffer);
    	GET_PROC(clEnqueueWriteBuffer);
    	GET_PROC(clGetContextInfo);
    	GET_PROC(clCreateProgramWithSource);
    	GET_PROC(clBuildProgram);
    	GET_PROC(clGetProgramBuildInfo);
    	GET_PROC(clCreateKernel);
    	GET_PROC(clSetKernelArg);
    	GET_PROC(clEnqueueNDRangeKernel);
    	GET_PROC(clFlush);
    	GET_PROC(clFinish);
    	GET_PROC(clRetainCommandQueue);
    	GET_PROC(clReleaseCommandQueue);
    	GET_PROC(clSetEventCallback);

    	return CL_SUCCESS;
    	//return CL_ERROR_UNKNOWN;
    }



extern int nbCudaDevices;

value spoc_clInit() {
	CAMLparam0();
	if (CL_SUCCESS != clInit())
	{

		noCL = 1;
 	}
	CAMLreturn(Val_unit);
}


#ifdef __cplusplus
}
#endif

