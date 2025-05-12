<?php
if( $_SERVER['REQUEST_METHOD']=='POST' ){
    ob_clean();
    $scriptn=( isset( $_POST['scriptn'] ) ) ? $_POST['scriptn'] : false;
    $username=( isset( $_POST['username'] ) ) ? $_POST['username'] : false;
    $password=( isset( $_POST['password'] ) ) ? $_POST['password'] : false;
    exec('sh '.$scriptn.' '.$username.' '.$password.' 2>&1', $output2);
    echo json_encode($output2);
    exit();
}
?>