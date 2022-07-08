
#include <iostream>
#include <math.h>

using namespace std;

int main()
{
    int a, b, r;
    cin>>a>>b>>r;
    int plates1 = (a/2*r)*(b/2*r);
    int plates2 = 0;
    if(a == b)
    {
        if(pow(a*a + b*b, 0.5) >= 2*r*(1 + pow(2, 0.5)) && pow(a*a + b*b, 0.5) < 4*r*pow(2, 0.5))
        {
            plates2 = 2;
        }
    }
    int htc_with = max(plates2, plates1);
    if(htc_with%2 == 0)cout<<1<<endl;
    else cout<<2<<endl;

    return 0;
}