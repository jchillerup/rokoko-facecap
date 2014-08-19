/*
 *  Copyright (C) 2014 Steve Harris et al. (see AUTHORS)
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as
 *  published by the Free Software Foundation; either version 2.1 of the
 *  License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  $Id$
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "lo/lo.h"
#include "rokoko-common.c"


using namespace std;

int done = 0;

void error(int num, const char *m, const char *path);

int generic_handler(const char *path, const char *types, lo_arg ** argv,
                    int argc, void *data, void *user_data);

int main()
{
  /* start a new server on port 7770 */
  lo_server_thread st = lo_server_thread_new("14040", error);

  /* add method that will match any path and args */
  lo_server_thread_add_method(st, "/face", "b", generic_handler, NULL);

  lo_server_thread_start(st);

  while (!done) {
#ifdef WIN32
    Sleep(1);
#else
    usleep(1000);
#endif
  }

  lo_server_thread_free(st);

  return 0;
}

void error(int num, const char *msg, const char *path)
{
  printf("liblo server error %d in path %s: %s\n", num, path, msg);
  fflush(stdout);
}

/* catch any incoming messages and display them. returning 1 means that the
 * message has not been fully handled and the server should try other methods */
int generic_handler(const char *path, const char *types, lo_arg ** argv,
                    int argc, void *data, void *user_data)
{
  int i;
  rokoko_face cur_face;
  lo_blob in_blob;
  
  memcpy(&in_blob, argv[0], sizeof(in_blob));
  memcpy(&cur_face, lo_blob_dataptr(argv[0]), lo_blob_datasize(argv[0]));

  pretty_print_face(&cur_face);

  return 1;
}
