#include <petscsys.h>

/*E
  WellFormedEnum - Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor

$ LOREM - A lorem
$ IPSUM - An ipsum
$ DOLOR - A dolor

  Level: advanced

.seealso: Lorem
E*/
typedef enum {
  LOREM,
  IPSUM,
  DOLOR
} WellFormedEnum;

/*
  IllFormedEnum -

$ SIT- A sit
$ CONSECTETUR - A consectetur
 $ AMET - An amet
$ADAPISCING - an adapiscing
Level: advanced
E*/
typedef enum {
  SIT,
  AMET,
  CONSECTETUR,
  ADAPISCING
} IllFormedEnum;
