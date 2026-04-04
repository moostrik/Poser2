from apps.hd_trio import HDTrioMain
# from apps.deep_flow import DeepFlowMain
# from apps.white_space import WhiteSpaceMain

APP_REGISTRY: dict[str, type] = {
    'hd_trio': HDTrioMain,
    # 'deep_flow': DeepFlowMain,
    # 'white_space': WhiteSpaceMain,
}
