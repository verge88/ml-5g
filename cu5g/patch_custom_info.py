from pathlib import Path


def replace_once(path, old, new):
    p=Path(path)
    text=p.read_text()
    if old not in text:
        raise SystemExit(f"stage39 customInfo patch marker missing in {path}")
    text=text.replace(old,new,1)
    p.write_text(text)


replace_once(
    "lib/sbi/context.h",
    "    char *fqdn;\n\n#define OGS_SBI_MAX_NUM_OF_IP_ADDRESS 8",
    "    char *fqdn;\n"
    "    /* Stage39: preserve schema-defined NFProfile.customInfo in NRF state. */\n"
    "    OpenAPI_object_t *custom_info;\n\n"
    "#define OGS_SBI_MAX_NUM_OF_IP_ADDRESS 8",
)

replace_once(
    "lib/sbi/context.c",
    "    nf_instance->fqdn = NULL;\n\n"
    "    for (i = 0; i < nf_instance->num_of_ipv4; i++) {",
    "    nf_instance->fqdn = NULL;\n\n"
    "    if (nf_instance->custom_info)\n"
    "        OpenAPI_object_free(nf_instance->custom_info);\n"
    "    nf_instance->custom_info = NULL;\n\n"
    "    for (i = 0; i < nf_instance->num_of_ipv4; i++) {",
)

replace_once(
    "lib/sbi/nnrf-handler.c",
    "    nf_instance->nf_type = NFProfile->nf_type;\n"
    "    nf_instance->nf_status = NFProfile->nf_status;\n"
    "    if (NFProfile->is_heart_beat_timer == true)",
    "    nf_instance->nf_type = NFProfile->nf_type;\n"
    "    nf_instance->nf_status = NFProfile->nf_status;\n"
    "    if (NFProfile->custom_info) {\n"
    "        cJSON *stage39_json =\n"
    "            OpenAPI_object_convertToJSON(NFProfile->custom_info);\n"
    "        if (stage39_json) {\n"
    "            nf_instance->custom_info =\n"
    "                OpenAPI_object_parseFromJSON(stage39_json);\n"
    "            cJSON_Delete(stage39_json);\n"
    "        }\n"
    "    }\n"
    "    if (NFProfile->is_heart_beat_timer == true)",
)

replace_once(
    "lib/sbi/nnrf-build.c",
    '#include "ogs-sbi.h"\n',
    '#include "ogs-sbi.h"\n'
    '#include <stdio.h>\n'
    '#include <stdlib.h>\n\n'
    'static OpenAPI_object_t *stage39_custom_info_from_env(void)\n'
    '{\n'
    '    const char *path = getenv("STAGE39_KIE_CUSTOM_INFO_PATH");\n'
    '    FILE *fp = NULL;\n'
    '    char buffer[32768];\n'
    '    size_t length = 0;\n'
    '    cJSON *json = NULL;\n'
    '    OpenAPI_object_t *object = NULL;\n\n'
    '    if (!path || !*path)\n'
    '        return NULL;\n'
    '    fp = fopen(path, "r");\n'
    '    if (!fp)\n'
    '        return NULL;\n'
    '    length = fread(buffer, 1, sizeof(buffer)-1, fp);\n'
    '    fclose(fp);\n'
    '    if (!length || length >= sizeof(buffer)-1)\n'
    '        return NULL;\n'
    '    buffer[length] = 0;\n'
    '    json = cJSON_Parse(buffer);\n'
    '    if (!json || !cJSON_IsObject(json)) {\n'
    '        if (json) cJSON_Delete(json);\n'
    '        return NULL;\n'
    '    }\n'
    '    object = OpenAPI_object_parseFromJSON(json);\n'
    '    cJSON_Delete(json);\n'
    '    return object;\n'
    '}\n\n',
)

replace_once(
    "lib/sbi/nnrf-build.c",
    "    NFProfile->nf_instance_id = nf_instance->id;\n"
    "    NFProfile->nf_type = nf_instance->nf_type;\n"
    "    NFProfile->nf_status = nf_instance->nf_status;\n",
    "    NFProfile->nf_instance_id = nf_instance->id;\n"
    "    NFProfile->nf_type = nf_instance->nf_type;\n"
    "    NFProfile->nf_status = nf_instance->nf_status;\n"
    "    if (nf_instance->custom_info) {\n"
    "        cJSON *stage39_json =\n"
    "            OpenAPI_object_convertToJSON(nf_instance->custom_info);\n"
    "        if (stage39_json) {\n"
    "            NFProfile->custom_info =\n"
    "                OpenAPI_object_parseFromJSON(stage39_json);\n"
    "            cJSON_Delete(stage39_json);\n"
    "        }\n"
    "    } else {\n"
    "        NFProfile->custom_info = stage39_custom_info_from_env();\n"
    "    }\n",
)

replace_once(
    "lib/sbi/nnrf-build.c",
    "    OpenAPI_patch_item_t StatusItem;\n"
    "    OpenAPI_patch_item_t LoadItem;\n",
    "    OpenAPI_patch_item_t StatusItem;\n"
    "    OpenAPI_patch_item_t LoadItem;\n"
    "    OpenAPI_patch_item_t Stage39CustomInfoItem;\n"
    "    OpenAPI_object_t *Stage39CustomInfo = NULL;\n"
    "    cJSON *Stage39CustomInfoJSON = NULL;\n",
)

replace_once(
    "lib/sbi/nnrf-build.c",
    "    memset(&StatusItem, 0, sizeof(StatusItem));\n"
    "    memset(&LoadItem, 0, sizeof(LoadItem));\n",
    "    memset(&StatusItem, 0, sizeof(StatusItem));\n"
    "    memset(&LoadItem, 0, sizeof(LoadItem));\n"
    "    memset(&Stage39CustomInfoItem, 0, sizeof(Stage39CustomInfoItem));\n",
)

replace_once(
    "lib/sbi/nnrf-build.c",
    "    OpenAPI_list_add(PatchItemList, &LoadItem);\n\n"
    "    message.PatchItemList = PatchItemList;\n",
    "    OpenAPI_list_add(PatchItemList, &LoadItem);\n\n"
    "    Stage39CustomInfo = stage39_custom_info_from_env();\n"
    "    if (Stage39CustomInfo) {\n"
    "        Stage39CustomInfoJSON =\n"
    "            OpenAPI_object_convertToJSON(Stage39CustomInfo);\n"
    "        if (!Stage39CustomInfoJSON) {\n"
    "            ogs_error(\"No Stage39 customInfo JSON\");\n"
    "            goto end;\n"
    "        }\n"
    "        Stage39CustomInfoItem.op = OpenAPI_patch_operation_replace;\n"
    "        Stage39CustomInfoItem.path = (char *)\"/customInfo\";\n"
    "        Stage39CustomInfoItem.value =\n"
    "            OpenAPI_any_type_create(Stage39CustomInfoJSON);\n"
    "        if (!Stage39CustomInfoItem.value) {\n"
    "            ogs_error(\"No Stage39 customInfo PatchItem value\");\n"
    "            goto end;\n"
    "        }\n"
    "        OpenAPI_list_add(PatchItemList, &Stage39CustomInfoItem);\n"
    "    }\n\n"
    "    message.PatchItemList = PatchItemList;\n",
)

replace_once(
    "lib/sbi/nnrf-build.c",
    "end:\n"
    "    if (LoadItem.value)\n"
    "        OpenAPI_any_type_free(LoadItem.value);\n",
    "end:\n"
    "    if (Stage39CustomInfoItem.value)\n"
    "        OpenAPI_any_type_free(Stage39CustomInfoItem.value);\n"
    "    if (Stage39CustomInfoJSON)\n"
    "        cJSON_Delete(Stage39CustomInfoJSON);\n"
    "    if (Stage39CustomInfo)\n"
    "        OpenAPI_object_free(Stage39CustomInfo);\n"
    "    if (LoadItem.value)\n"
    "        OpenAPI_any_type_free(LoadItem.value);\n",
)

replace_once(
    "src/nrf/nnrf-handler.c",
    "            CASE(OGS_SBI_PATCH_PATH_PLMN_LIST)\n",
    "            CASE(\"/customInfo\")\n"
    "                if (!patch_item->value || !patch_item->value->json ||\n"
    "                    !cJSON_IsObject(patch_item->value->json)) {\n"
    "                    ogs_assert(true == ogs_sbi_server_send_error(\n"
    "                        stream, OGS_SBI_HTTP_STATUS_BAD_REQUEST, recvmsg,\n"
    "                        \"Invalid value for /customInfo\", NULL, NULL));\n"
    "                    return false;\n"
    "                }\n"
    "                if (nf_instance->custom_info)\n"
    "                    OpenAPI_object_free(nf_instance->custom_info);\n"
    "                nf_instance->custom_info = NULL;\n"
    "                if (patch_item->value->json->child) {\n"
    "                    nf_instance->custom_info = OpenAPI_object_parseFromJSON(\n"
    "                        patch_item->value->json);\n"
    "                    if (!nf_instance->custom_info) {\n"
    "                        ogs_assert(true == ogs_sbi_server_send_error(\n"
    "                            stream, OGS_SBI_HTTP_STATUS_BAD_REQUEST,\n"
    "                            recvmsg, \"Cannot parse /customInfo\", NULL, NULL));\n"
    "                        return false;\n"
    "                    }\n"
    "                }\n"
    "                break;\n"
    "            CASE(OGS_SBI_PATCH_PATH_PLMN_LIST)\n",
)

print("STAGE39_NFPROFILE_CUSTOM_INFO_PATCH_OK")
