#include "testheader.h"

void BareFunctionShouldGetStatic(void) { }

extern void ExternFunctionShouldNotGetStatic(void) { }

static void StaticFunctionShouldNotGetStatic(void) { }

// this should not get static
static void StaticFunctionPreDeclShouldNotGetStatic(void);

// this should get static!
void StaticFunctionPreDeclShouldNotGetStatic(void) { }

extern void ExternFunctionPreDeclShouldNotGetStatic(void);

void ExternFunctionPreDeclShouldNotGetStatic(void) { }

void BareFunctionPreDeclShouldGetStatic(void);

void BareFunctionPreDeclShouldGetStatic(void) { }

// declaration in testheader has "extern"
void ExternHeaderFunctionShouldNotGetStatic(void) { }

class Foo {
public:
  friend void swap();
};

void swap() { }

// clang-format off
void                                                    testBadFormatting                      ( void)
{

}
// clang-format on

// ironically enough, this will get static
void silence_warnings(void)
{
  (void)StaticFunctionShouldNotGetStatic;
  (void)StaticFunctionPreDeclShouldNotGetStatic;
}
