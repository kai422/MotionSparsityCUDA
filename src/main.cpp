#include <torch/torch.h>
#include <iostream>
#include <bitset>
#include <vector>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

int main() {

    /*
     torch::Tensor tensor = torch::rand({2, 3});
     std::cout << tensor.size(0) << std::endl;
     std::cout << __cplusplus << std::endl;
    for(auto& i : {3,4,5,6}){
        std::cout << i <<' ';
    }
     */
    /*
    class Classtype
    {
    public:
        Classtype(int a, int b, int c):k(a),j(b)
        {
            m=k+j+c;
            std::cout << "Hello Clastype!" << k<<j<<m << std::endl;
        }
        bool operator ()(int b)
        {
            std::cout << "Hello Clastype()!" << b << std::endl;
            return true;
        }
    private:
        int k;
        int j;
        int m;
    };
    Classtype(1,2,0)(88);
    Classtype a(77,12,12);
    std::cout << a(56);
     */
    /*
    int* a = new int[3]{};
    for(auto i : {0,1,2})
    {
        std::cout<<a[i]<<std::endl;
    }
     */
    /*int* num = new int[2]{0,0};
    int pos = 1;
    const int N_QUAD_TREE_T_BITS = 8*sizeof(int);
    std::cout<<pos / N_QUAD_TREE_T_BITS<<std::endl;
    std::cout<<pos % N_QUAD_TREE_T_BITS<<std::endl;
    num[pos / N_QUAD_TREE_T_BITS] |= (1 << (pos % N_QUAD_TREE_T_BITS));
    for(auto i : {0,1})
    {
        std::cout<<num[i]<<std::endl;
    }
     */
    /*
    bitset<21>* trees = new bitset<21>[3]{};
    for(auto i : {0,1,2})
    {
        std::cout<<trees[i]<<std::endl;
    }
    trees[1].set(3);
    trees[1].set(5);
    for(auto i : {0,1,2})
    {
        std::cout<<trees[i]<<std::endl;
    }
     */
    /*
    class foo{
    public:
        foo()
        {
                array = new int[3]{1,2,3};
        }
        int *array;
        int* getarray() const
        {
            return array;
        }
    };

    foo i{};
    int * array_=i.getarray();
    array_[0]=3;
    int& array_0=array_[0];
    array_0=1;
    for(auto i : {0,1,2})
    {
        std::cout<<array_[i]<<std::endl;
    }
    */
    /*
    class foo{
    public:
        foo()
        {
        }
        vector<int> array;
    };

    foo o;
    std::cout<<o.array.empty()<<std::endl;
    o.array.push_back(12);
    std::cout<<o.array.data()<<std::endl;
    o.array.push_back(13);
    std::cout<<o.array.data()<<std::endl;
    std::cout<<o.array.size()<<std::endl;
    for(auto i:o.array)
    {
        std::cout<<i<<std::endl;
    }
     */
    torch::Tensor t = torch::rand({2, 3, 2});
    cout << t.numel() << endl;
    auto t_d = t.data_ptr<float>();
    cout << t_d[1] << endl;
    /*int a=0;
    float b=0.f;
    cout<<(a!=b)<<endl;
     */
    /*auto t1 = t[1];

    cout<<t1<<endl;
    cout<<t<<endl;
    auto data = t1.template data_ptr<float>();
    data[2]=0.422;

    cout<<t1<<endl;
    cout<<t<<endl;
*/
    /*
    torch::Tensor t1 = torch::rand({2, 3, 2});

    auto input_sizes = t.sizes();
    auto input_sizes1 = t1.sizes();
    std::cout << (input_sizes==input_sizes1) << std::endl;
    */
    /*std::cout << t << std::endl;
   auto data = t.template data_ptr<float>();
   auto data_a = t.accessor<float,3>();
   std::cout << t.size(1) << std::endl;
   //x*dim1+y
   for(auto i :{0,1,2,3,4,5})
   {
       std::cout << data[i] << std::endl;
   }
   std::cout << data[1*(t.size(1)*t.size(2))+2*t.size(2)+1] << std::endl;
   std::cout << data_a[1][2][1] << std::endl;
    */
    /*
    auto t_a = t.accessor<float,2>();

    std::cout << t_a[0][0] << std::endl;
    std::cout << t_a[0][1] << std::endl;
    std::cout << t_a[0][2] << std::endl;
    std::cout << t_a[1][0] << std::endl;
    std::cout << t_a[1][1] << std::endl;
    std::cout << t_a[1][2] << std::endl;
    std::cout << t_a[0][5] << std::endl;

    int a[2][3]{1,2,3,4,5,6};
    std::cout << a[0][3] << std::endl;
    */
    /*
    torch::Tensor in = torch::rand({3, 2, 2});
    //std::cout << in.dim()  << std::endl;
    //std::cout << in.ndimension()  << std::endl;
    //auto s = in.sizes();
    //std::cout << s << std::endl;
    auto input_t = in[0];
    std::cout << input_t << std::endl;
    std::cout << input_t.type() << std::endl;
    */
    /*
    bitset<3> x = bitset<3>();
    std::cout << !x[0]+!x[1]+!x[2] << std::endl;
    */
    /*
    //cv::Mat a =

    class Create {
    public:
        Create(int x, int y) {
            cout<<"ctor"<<endl;
            h = new int[3]{x,y,0};
        }

        ~Create()
        {
            cout<<"dtor"<<endl;
        }
        int* h;
    };
    int* h_;
    {
        Create create(1,2);
        h_=create.h;
    }
    cout<<h_[0]<<endl;
    */
    /*
    int* ptr_ptr[3];
    ptr_ptr[0]=new int[3]{0,1,2};
    ptr_ptr[1]=new int[3]{5,6,7};
    cout<<*(*(ptr_ptr+1)+2)<<endl;
     */

}
