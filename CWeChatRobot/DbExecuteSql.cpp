#include "pch.h"

// ����DLL�ӿ�ʱ�Ĳ���
struct executeParams {
	DWORD ptrDb;
	DWORD ptrSql;
};

// ����DLL�ķ������ݣ�������̬�����׵�ַ�����鳤��
struct executeResult {
	DWORD SQLResultData;
	DWORD length;
};

// ����ReadProcessMemory�����Ľṹ��
struct SQLResultAddrStruct {
	DWORD ColName;
	DWORD l_ColName;
	DWORD content;
	DWORD l_content;
};

// vector�����ݽṹ
struct VectorStruct {
#ifdef _DEBUG
	DWORD v_head;
#endif
	DWORD v_data;
	DWORD v_end1;
	DWORD v_end2;
};

// ����SQL��ѯ�ṹ�Ļ����ṹ
struct SQLResultStruct {
	wchar_t* ColName;
	wchar_t* content;
};
// ��ѯ�����һ����ά����
vector<vector<SQLResultStruct>> SQLResult;

// ÿ�β�ѯǰ���ǰһ�β�ѯ���Ľ��
void ClearResultArray() {
	if (SQLResult.size() == 0)
		return;
	for (unsigned int i = 0; i < SQLResult.size(); i++) {
		for (unsigned j = 0; j < SQLResult[i].size(); j++) {
			SQLResultStruct* sr = (SQLResultStruct*)&SQLResult[i][j];
			if (sr->ColName) {
				delete sr->ColName;
				sr->ColName = NULL;
			}
			if (sr->content) {
				delete sr->content;
				sr->content = NULL;
			}
		}
		SQLResult[i].clear();
	}
	SQLResult.clear();
}

// ������ѯ���������SAFEARRAY
SAFEARRAY* CreateSQLResultSafeArray() {
	if (SQLResult.size() == 0 || SQLResult[0].size() == 0)
		return NULL;
	SAFEARRAYBOUND rgsaBound[2] = { {SQLResult.size() + 1,0},{SQLResult[0].size(),0}};
	SAFEARRAY* psaValue = SafeArrayCreate(VT_VARIANT, 2, rgsaBound);
	HRESULT hr = S_OK;
	long Index[2] = { 0,0 };
	for (unsigned int i = 0; i < SQLResult.size(); i++) {
		for (unsigned int j = 0; j < SQLResult[i].size(); j++) {
			SQLResultStruct* ptrResult = (SQLResultStruct*)&SQLResult[i][j];
			if (i == 0)
			{
				Index[0] = 0; Index[1] = j;
				hr = SafeArrayPutElement(psaValue, Index, &(_variant_t)ptrResult->ColName);
			}
			Index[0] = i + 1; Index[1] = j;
			hr = SafeArrayPutElement(psaValue, Index, &(_variant_t)ptrResult->content);
		}
	}
	return psaValue;
}

// ������ѯ���
VOID ReadSQLResultFromWeChatProcess(DWORD dwHandle) {
	executeResult result = { 0 };
	ReadProcessMemory(hProcess, (LPCVOID)dwHandle, &result, sizeof(executeResult), 0);
	for (unsigned int i = 0; i < result.length; i++) {
		VectorStruct v_temp = { 0 };
		vector<SQLResultStruct> s_temp;
		ReadProcessMemory(hProcess, (LPCVOID)result.SQLResultData, &v_temp, sizeof(VectorStruct), 0);
		while (v_temp.v_data < v_temp.v_end1) {
			SQLResultAddrStruct sqlresultAddr = { 0 };
			SQLResultStruct sqlresult = { 0 };
			ReadProcessMemory(hProcess, (LPCVOID)v_temp.v_data, &sqlresultAddr, sizeof(SQLResultAddrStruct), 0);
			char* ColName = new char[sqlresultAddr.l_ColName + 1];
			sqlresult.ColName = new wchar_t[sqlresultAddr.l_ColName + 1];
			ReadProcessMemory(hProcess, (LPCVOID)sqlresultAddr.ColName, ColName, sqlresultAddr.l_ColName + 1, 0);
			MultiByteToWideChar(CP_ACP,0,ColName,-1,sqlresult.ColName,strlen(ColName) + 1);
			char* content = new char[sqlresultAddr.l_content + 1];
			sqlresult.content = new wchar_t[sqlresultAddr.l_content + 1];
			ReadProcessMemory(hProcess, (LPCVOID)sqlresultAddr.content, content, sqlresultAddr.l_content + 1, 0);
			MultiByteToWideChar(CP_UTF8, 0, content, -1, sqlresult.content, strlen(content) + 1);
			delete[] ColName;
			ColName = NULL;
			delete[] content;
			content = NULL;
			v_temp.v_data += sizeof(SQLResultAddrStruct);
			s_temp.push_back(sqlresult);
		}
		SQLResult.push_back(s_temp);
		result.SQLResultData += sizeof(VectorStruct);
	}
}

SAFEARRAY* ExecuteSQL(DWORD DbHandle,BSTR sql) {
	if (!hProcess)
		return NULL;
	ClearResultArray();
	DWORD dwHandle = 0x0;
	DWORD dwId = 0x0;
	DWORD dwWriteSize = 0x0;
	LPVOID sqlAddr = VirtualAllocEx(hProcess, NULL, 1, MEM_COMMIT, PAGE_READWRITE);
	executeParams* paramAndFunc = (executeParams*)::VirtualAllocEx(hProcess, 0, sizeof(executeParams), MEM_COMMIT, PAGE_READWRITE);
	if (!sqlAddr || !paramAndFunc)
		return NULL;
	char* a_sql = _com_util::ConvertBSTRToString(sql);
	if(sqlAddr)
		WriteProcessMemory(hProcess, sqlAddr, a_sql, strlen(a_sql) + 1, &dwWriteSize);
	executeParams param = { 0 };
	param.ptrDb = DbHandle;
	param.ptrSql = (DWORD)sqlAddr;

	if(paramAndFunc)
		WriteProcessMemory(hProcess, paramAndFunc, &param, sizeof(executeParams), &dwWriteSize);

	DWORD ExecuteSQLRemoteAddr = GetWeChatRobotBase() + ExecuteSQLRemoteOffset;
	HANDLE hThread = ::CreateRemoteThread(hProcess, NULL, 0, (LPTHREAD_START_ROUTINE)ExecuteSQLRemoteAddr, (LPVOID)paramAndFunc, 0, &dwId);
	if (hThread) {
		WaitForSingleObject(hThread, INFINITE);
		GetExitCodeThread(hThread, &dwHandle);
		CloseHandle(hThread);
	}
	else {
		return NULL;
	}
	if (!dwHandle)
		return NULL;
	ReadSQLResultFromWeChatProcess(dwHandle);
	SAFEARRAY* psaValue = CreateSQLResultSafeArray();
	VirtualFreeEx(hProcess, sqlAddr, 0, MEM_RELEASE);
	VirtualFreeEx(hProcess, paramAndFunc, 0, MEM_RELEASE);
	return psaValue;
}