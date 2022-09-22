#include "memory.h"

//Binary patch addresses
DWORD patch_main_map = 0x005124E6;

const int total_addresses = 1;
const int patch_size = 2; //patch bytes
bool patched = false;

struct memory_alt {
	DWORD patch_address;
	byte patch_bytes[patch_size];
	byte oringinal_bytes[patch_size];
};

struct memory_alt memory_alts[total_addresses];

void init_binary_patch() {
	memory_alts[0] = { patch_main_map, {0xeb, 0x14}, {00}};
	for (int i=0; i<total_addresses; i++) {
		readBytes((void*)memory_alts[i].patch_address, &memory_alts[i].oringinal_bytes, patch_size);
}
}

void toggle_patch() {
	void* to_patch;
	for (int i; i<total_addresses, i++) {
		if (patched) {
			to_patch = &memory_alts[i].oringinal_bytes;
		} else {
			to_patch = &memory_alts[i].patch_bytes;
		}
		writeBytes((void*)memory_alts[i].patch_address, to_patch, patch_size);
	}
	patched = !patched;QDOgJAvG9bQ
}





//https://www.youtube.com/watch?v=