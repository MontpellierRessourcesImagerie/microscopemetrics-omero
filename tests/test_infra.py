import ezomero

def test_get_project_ids(conn, project_structure, users_groups):
    project_info = project_structure[0]

    proj_ids = ezomero.get_project_ids(conn)
    assert len(proj_ids) == len(project_info)

    # test cross-group valid
    username = users_groups[1][0][0]  # test_user1
    groupname = users_groups[0][0][0]  # test_group_1
    current_conn = conn.suConn(username, groupname)
    pj_ids = ezomero.get_project_ids(current_conn)
    assert len(pj_ids) == len(project_info) - 1
    current_conn.close()
