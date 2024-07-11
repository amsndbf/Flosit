module FLOSIT_mult_clk #(  
    parameter N = 32,
    parameter OUT_WIDTH = 6 
    )  
(	
input rst,
input Clk,
input F_P,
input [N-1:0] in1,
input [N-1:0] in2,
output reg[N-1:0] out_reg,
output reg P_inf_reg,
output reg P_zero_reg,
output reg done_reg,
output reg [OUT_WIDTH-1:0] F_Exceptions_reg,
input start
);
    
parameter Bs = 5;//log2(N)
parameter es = 2;//es

reg [N-1:0]in1_reg, in2_reg;
reg F_P_reg, start_reg;
wire [N-1:0] P_out, out;
wire P_inf, P_zero;
wire [OUT_WIDTH-1:0]F_Exceptions;
wire done;

FLOSIT_mult #(.N(N),.Bs(Bs), .es(es)) FLOSIT_mult_1(.F_P(F_P_reg), .in1(in1_reg), .in2(in2_reg), .start(start_reg), .out(out), .P_inf(P_inf), .P_zero(P_zero), .done(done), .F_Exceptions(F_Exceptions));
always @(posedge Clk ) begin
if (rst) begin
    F_P_reg             <= 0;
	in1_reg		        <= 0;
	in2_reg		        <= 0;
    start_reg	        <= 0;

    out_reg 	        <= 0;
    P_inf_reg 	        <= 0;
    P_zero_reg	        <= 0;
    F_Exceptions_reg    <= 0;
    done_reg 	        <= 0;
end else begin
    F_P_reg             <= F_P;
	in1_reg		        <= in1;
	in2_reg		        <= in2;
    start_reg	        <= start;

    out_reg 	        <= out;
    P_inf_reg 	        <= P_inf;
    P_zero_reg	        <= P_zero;
    F_Exceptions_reg    <= F_Exceptions;
    done_reg 	        <= done;
end

end	
endmodule

module FLOSIT_mult(F_P, in1, in2, start, out, P_inf, P_zero, done, F_Exceptions);

parameter N = 32;//N
parameter Bs = 5;//log2(N)
parameter es = 2;
parameter E = 8;

input [N-1:0] in1, in2;
input start;
input [2:0] F_P; 
output [N-1:0] out;
output P_inf, P_zero;
output done;
output [5:0]F_Exceptions;
wire [N-1:0] P_out;

wire [N-1:0] in_1_P, in_2_P, in_1_F, in_2_F;

FP_to_posit #(.N(N), .E(E), .es(es)) FPC1(F_P[0], in1, in_1_P);
FP_to_posit #(.N(N), .E(E), .es(es)) FPC2(F_P[1], in2, in_2_P);
Posit_to_FP #(.N(N), .E(E), .es(es)) PFC1(F_P[0], in1, in_1_F);
Posit_to_FP #(.N(N), .E(E), .es(es)) PFC2(F_P[1], in2, in_2_F);


//__________________________________________________________
// Posit --> inf & zero
wire start0= start;
wire s1 = in_1_P[N-1];
wire s2 = in_2_P[N-1];
wire zero_tmp1 = |in_1_P[N-2:0];
wire zero_tmp2 = |in_2_P[N-2:0];
wire inf1 = in_1_P[N-1] & (~zero_tmp1),
	inf2 = in_2_P[N-1] & (~zero_tmp2);
wire zero1 = ~(in_1_P[N-1] | zero_tmp1),
	zero2 = ~(in_2_P[N-1] | zero_tmp2);
assign P_inf = inf1 | inf2,
	P_zero = zero1 & zero2;
//__________________________________________________________
// IEEE754 --> Data Extraction
	wire [7:0]  FA_E, FB_E, FS_E;
    wire [22:0] FA_F, FB_F, FS_F;
    wire FA_S, FB_S;
    //Multiply by OP
    //assign FB_M = {op_reg^FB_reg[31],FB_reg[30:0]};
	assign {FA_S, FA_E, FA_F} = in_1_F ;
	assign {FB_S, FB_E, FB_F} = in_2_F ;
    
    //extend  {Sign, carry, hidden}
    wire [23:0]FA_F_ext, FB_F_ext;
    assign FA_F_ext = (FA_E != 8'd0) ? {1'b1, FA_F}: {1'b0, FA_F}; // hiden bit exist or not
    assign FB_F_ext = (FB_E != 8'd0) ? {1'b1, FB_F}: {1'b0, FB_F}; // hiden bit exist or not
//__________________________________________________________
// Posit --> Data Extraction
wire rc1, rc2;
wire [Bs-1:0] regime1, regime2;
wire [es-1:0] e1, e2;
wire [N-es-1:0] mant1, mant2;
wire [N-1:0] xin1 = s1 ? -in_1_P : in_1_P;
wire [N-1:0] xin2 = s2 ? -in_2_P : in_2_P;
data_extract_v1 #(.N(N),.es(es)) uut_de1(.in(xin1), .rc(rc1), .regime(regime1), .exp(e1), .mant(mant1));
data_extract_v1 #(.N(N),.es(es)) uut_de2(.in(xin2), .rc(rc2), .regime(regime2), .exp(e2), .mant(mant2));
//FLOSIT _____________________________________________________________________________________________________
wire [N-es:0] 	m1 = (F_P[2])?{7'd0, FA_F_ext}:{zero_tmp1, mant1}, 
                m2 = (F_P[2])?{7'd0, FB_F_ext}:{zero_tmp2, mant2};
//wire [N-es:0] m1 = {zero_tmp1,mant1}, 
//				m2 = {zero_tmp2,mant2};
//__________________________________________________________
// IEEE754 --> Exceptions
	wire Fr_zero_A, Unlimited_A, NaN_A;
    wire Fr_zero_B, Unlimited_B, NaN_B;
    wire Fr_zero, Unlimited, NaN;
    assign Fr_zero_A      = (FA_F == 23'd0)                         ? 1'b1 : 1'b0 ;
    assign Unlimited_A    = ((FA_E == 8'b1111_1111) & Fr_zero_A)    ? 1'b1 : 1'b0 ; //unlimited
    assign NaN_A          = ((FA_E == 8'b1111_1111) & !Fr_zero_A)   ? 1'b1 : 1'b0 ; //NaN
    assign Fr_zero_B      = (FB_F == 23'd0)                       	? 1'b1 : 1'b0 ;
    assign Unlimited_B    = ((FB_E == 8'b1111_1111) & Fr_zero_B)    ? 1'b1 : 1'b0 ; //unlimited
    assign NaN_B          = ((FB_E == 8'b1111_1111) & !Fr_zero_B)   ? 1'b1 : 1'b0 ; //NaN
    assign F_Exceptions = { Unlimited, NaN, Unlimited_A, NaN_A, Unlimited_B, NaN_B }; // Exceptions wire 

//__________________________________________________________
//Posit --> Sign, Exponent and Mantissa Computation
	wire mult_s = s1 ^ s2;

	wire [2*(N-es)+1:0] mult_m = m1*m2;
	//FLOSIT _____________________________________________________________________________________________________
	wire mult_m_ovf = (F_P[2]) ? (1'b0) : (mult_m[2*(N-es)+1]);
	//wire mult_m_ovf = mult_m[2*(N-es)+1];
	wire [2*(N-es)+1:0] mult_mN = ~mult_m_ovf ? mult_m << 1'b1 : mult_m;

	wire [Bs+1:0] r1 = rc1 ? {2'b0,regime1} : -regime1;
	wire [Bs+1:0] r2 = rc2 ? {2'b0,regime2} : -regime2;
	wire [Bs+es+1:0] mult_e;
	//FLOSIT _____________________________________________________________________________________________________
	add_N_Cin #(.N(Bs+es+1)) uut_add_exp ((F_P[2])?(FA_E):({r1,e1}), (F_P[2])?(FB_E):({r2,e2}), mult_m_ovf, mult_e);
	//add_N_Cin #(.N(Bs+es+1)) uut_add_exp ({r1,e1}, {r2,e2}, mult_m_ovf, mult_e);
//__________________________________________________________
// IEEE754 --> Exponent
	wire [47:0]FS_F_ext,FS_F_END;
	assign FS_F_ext = mult_m[47:0];//[61:14] 62 = 2*(N-es)+1, 14 = 62-48

//__________________________________________________________
// IEEE754 --> Sign_bit
  wire FS_S;
	assign FS_S = FA_S ^ FB_S; //sign_bit
//__________________________________________________________
// IEEE754 --> To Final
  wire [N-1:0]FS;
  wire [5:0] FS_shift_num;
	PENC64 P0(.Din({16'd0, FS_F_ext[47:0]}), .Dout(FS_shift_num), .valid(valid));
	assign FS_E = ((FA_E == 8'd0)&(FB_E == 8'd0)) ? FA_E : (mult_e-8'd127 - (6'd46 - FS_shift_num));//E_bit
	//assign FS_E = ((FA_E == 8'd0)&(FB_E == 8'd0)) ? FA_E : (op)?(((FA_E>FB_E)?(FA_E-FB_E):(FB_E-FA_E))- (6'd46 - FS_shift_num)):( ((FA_E - 8'd127) + FB_E ) - (6'd46 - FS_shift_num));//E_bit
	assign FS_F_END = ((FA_E == 8'd0)&(FB_E == 8'd0)) ? ( FS_F_ext[47:0] ) : ( FS_F_ext[47:0] << (6'd48 - FS_shift_num));//F_bit

	//Non-normal-signal
	assign Fr_zero      = (FS_F_END[47:25] == 23'd0)          	? 1'b1 : 1'b0 ;
	assign Unlimited    = ((FS_E == 8'b1111_1111) & Fr_zero)    ? 1'b1 : 1'b0 ;    //unlimited
	assign NaN          = ((FS_E == 8'b1111_1111) & !Fr_zero)   ? 1'b1 : 1'b0 ;         //NaN

	//zero det
	assign zero = !(1);                                            //[BAD]*_* --> (1)
	assign FS = (zero)? 32'd0 : {FS_S, FS_E, FS_F_END[47:26], |FS_F_END[25:0]};
//__________________________________________________________
//Posit --> Exponent and Regime Computation
	wire [es-1:0] e_o;
	wire [Bs:0] r_o;
	reg_exp_op #(.es(es), .Bs(Bs)) uut_reg_ro (mult_e[es+Bs+1:0], e_o, r_o);

//__________________________________________________________
//Posit --> Exponent, Mantissa and GRS Packing
	wire [2*N-1+3:0]tmp_o = {{N{~mult_e[es+Bs+1]}},mult_e[es+Bs+1],e_o,mult_mN[2*(N-es):2*(N-es)-(N-es-1)+1], mult_mN[2*(N-es)-(N-es-1):2*(N-es)-(N-es-1)-1], |mult_mN[2*(N-es)-(N-es-1)-2:0] }; 


//__________________________________________________________
//Posit --> Including Regime bits in Exponent-Mantissa Packing
	wire [3*N-1+3:0] tmp1_o;
	DSR_right_N_S #(.N(3*N+3), .S(Bs+1)) dsr2 (.a({tmp_o,{N{1'b0}}}), .b(r_o[Bs] ? {Bs{1'b1}} : r_o), .c(tmp1_o));

//__________________________________________________________
//Posit --> Rounding RNE : ulp_add = G.(R + S) + L.G.(~(R+S))
	wire 	L = tmp1_o[N+4], G = tmp1_o[N+3], R = tmp1_o[N+2], St = |tmp1_o[N+1:0],
     		ulp = ((G & (R | St)) | (L & G & ~(R | St)));
	wire [N-1:0] rnd_ulp = {{N-1{1'b0}},ulp};

	wire [N:0] tmp1_o_rnd_ulp;
	add_N #(.N(N)) uut_add_ulp (tmp1_o[2*N-1+3:N+3], rnd_ulp, tmp1_o_rnd_ulp);
	wire [N-1:0] tmp1_o_rnd = (r_o < N-es-2) ? tmp1_o_rnd_ulp[N-1:0] : tmp1_o[2*N-1+3:N+3];


//__________________________________________________________
//Posit --> Final Output
	wire [N-1:0] tmp1_oN = mult_s ? -tmp1_o_rnd : tmp1_o_rnd;
	assign 	P_out = P_inf|P_zero|(~mult_mN[2*(N-es)+1]) ? {P_inf,{N-1{1'b0}}} : {mult_s, tmp1_oN[N-1:1]},
			done = start0;

//________________________________________________________________________________________________
//FLOSIT -->  Final Output			
	assign out = (F_P[2]) ? (FS) : (P_out);


endmodule
//____________________________________________________________________________________
// =============== POSIT module =======================
//____________________________________________________________________________________
/////////////////////////
module data_extract_v1(in, rc, regime, exp, mant);

/* function [31:0] log2;
input reg [31:0] value;
	begin
	value = value-1;
	for (log2=0; value>0; log2=log2+1)
        	value = value>>1;
      	end
endfunction */

parameter N=32;//N
parameter Bs=5;//log2(N)
parameter es = 2;
input [N-1:0] in;
output rc;
output [Bs-1:0] regime;
output [es-1:0] exp;
output [N-es-1:0] mant;

wire [N-1:0] xin = in;
assign rc = xin[N-2];

wire [N-1:0] xin_r = rc ? ~xin : xin;

wire [Bs-1:0] k;
LOD_N #(.N(N)) xinst_k(.in({xin_r[N-2:0],rc^1'b0}), .out(k));

assign regime = rc ? k-1 : k;

wire [N-1:0] xin_tmp;
DSR_left_N_S #(.N(N), .S(Bs)) ls (.a({xin[N-3:0],2'b0}),.b(k),.c(xin_tmp));

assign exp= xin_tmp[N-1:N-es];
assign mant= xin_tmp[N-es-1:0];

endmodule


/////////////////
module sub_N (a,b,c);
parameter N=10;
input [N-1:0] a,b;
output [N:0] c;
assign c = {1'b0,a} - {1'b0,b};
endmodule

/////////////////////////
module add_N (a,b,c);
parameter N=10;
input [N-1:0] a,b;
output [N:0] c;
assign c = {1'b0,a} + {1'b0,b};
endmodule

/////////////////////////
module add_N_Cin (a,b,cin,c);
parameter N=10;
input [N:0] a,b;
input cin;
output [N:0] c;
assign c = a + b + cin;
endmodule


/////////////////////////
module add_1 (a,mant_ovf,c);
parameter N=10;
input [N:0] a;
input mant_ovf;
output [N:0] c;
assign c = a + mant_ovf;
endmodule

/////////////////////////
module conv_2c (a,c);
parameter N=10;
input [N:0] a;
output [N:0] c;
assign c = a + 1'b1;
endmodule

/////////////////////////
module reg_exp_op (exp_o, e_o, r_o);
parameter es=3;
parameter Bs=5;
input [es+Bs+1:0] exp_o;
output [es-1:0] e_o;
output [Bs:0] r_o;

assign e_o = exp_o[es-1:0];

wire [es+Bs:0] exp_oN_tmp;
conv_2c #(.N(es+Bs)) uut_conv_2c1 (~exp_o[es+Bs:0],exp_oN_tmp);
wire [es+Bs:0] exp_oN = exp_o[es+Bs+1] ? exp_oN_tmp[es+Bs:0] : exp_o[es+Bs:0];

assign r_o = (~exp_o[es+Bs+1] || |(exp_oN[es-1:0])) ? exp_oN[es+Bs:es] + 1 : exp_oN[es+Bs:es];
endmodule

/////////////////////////
module DSR_left_N_S(a,b,c);
        parameter N= 32;//N
        parameter S= 5;//log2(N)
        input [N-1:0] a;
        input [S-1:0] b;
        output [N-1:0] c;

//wire [N-1:0] tmp [S-1:0];
reg [N-1:0] tmp [S-1:0];
//assign tmp[0]  = b[0] ? a << 7'd1  : a; 
integer i;
always @(*) begin
	tmp[0]  <= b[0] ? a << 7'd1  : a; 
	for (i=1; i<S; i=i+1)begin
		tmp[i] <= b[i] ? tmp[i-1] << 2**i : tmp[i-1];
	end
end
/* genvar i;
generate
	begin 
		for (i=1; i<S; i=i+1)begin:loop_blk
			assign tmp[i] = b[i] ? tmp[i-1] << 2**i : tmp[i-1];
		end
	end
endgenerate */
assign c = tmp[S-1];

endmodule



/////////////////////////
module DSR_right_N_S(a,b,c);
        parameter N= 32;//N
        parameter S= 5;//log2(N)
        input [N-1:0] a;
        input [S-1:0] b;
        output [N-1:0] c;

//wire [N-1:0] tmp [S-1:0];
reg [N-1:0] tmp [S-1:0];
//assign tmp[0]  = b[0] ? a >> 7'd1  : a; 
integer i;
always @(*) begin
	tmp[0]  <= b[0] ? a >> 7'd1  : a; 
	for (i=1; i<S; i=i+1)begin
		tmp[i] <= b[i] ? tmp[i-1] >> 2**i : tmp[i-1];
	end
end
/* genvar i;
generate
	begin 
		for (i=1; i<S; i=i+1)begin:loop_blk
			assign tmp[i] = b[i] ? tmp[i-1] >> 2**i : tmp[i-1];
		end
	end
endgenerate */
assign c = tmp[S-1];

endmodule

/////////////////////////

module LOD_N (in, out);

/*   function [31:0] log2;
    input reg [31:0] value;
    begin
      value = value-1;
      for (log2=0; value>0; log2=log2+1)
	value = value>>1;
    end
  endfunction */

parameter N = 32;//N
parameter S = 5;//log2(N) 
input [N-1:0] in;
output [S-1:0] out;

wire vld;
LOD_32  l1 (in, out, vld);
endmodule
/* 
module LOD (in, out, vld);
function [31:0] log2;
input reg [31:0] value;
begin
  value = value-1;
  for (log2=0; value>0; log2=log2+1)
value = value>>1;
end
endfunction

parameter N = 8;
parameter S = log2(N);

   input [N-1:0] in;
   output [S-1:0] out;
   output vld;
  generate
	begin 
		if (N == 2)
		begin
	  assign vld = |in;
	  assign out = ~in[1] & in[0];
		end
	  else if (N & (N-1))
		//LOD #(1<<S) LOD ({1<<S {1'b0}} | in,out,vld);
		LOD #(1<<S) LOD ({in,{((1<<S) - N) {1'b0}}},out,vld);
	  else
		begin
	  wire [S-2:0] out_l, out_h;
	  wire out_vl, out_vh;
	  LOD #(N>>1) l(in[(N>>1)-1:0],out_l,out_vl);
	  LOD #(N>>1) h(in[N-1:N>>1],out_h,out_vh);
	  assign vld = out_vl | out_vh;
	  assign out = out_vh ? {1'b0,out_h} : {out_vl,out_l};
		end
	end
  endgenerate
endmodule */
module LOD_32 (in, out, vld);

parameter N = 32;
parameter S = 5;

   input [N-1:0] in;
   output [S-1:0] out;
   output vld;
   //
   wire [S-2:0] out_l, out_h;
   wire out_vl, out_vh;
   LOD_16 #(N>>1, S-1) l(in[(N>>1)-1:0],out_l,out_vl);
   LOD_16 #(N>>1, S-1) h(in[N-1:N>>1],out_h,out_vh);
   assign vld = out_vl | out_vh;
   assign out = out_vh ? {1'b0,out_h} : {out_vl,out_l};
   //

endmodule

module LOD_16 (in, out, vld);

parameter N = 32;
parameter S = 5;

   input [N-1:0] in;
   output [S-1:0] out;
   output vld;
   //
   wire [S-2:0] out_l, out_h;
   wire out_vl, out_vh;
   LOD_8 #(N>>1, S-1) l(in[(N>>1)-1:0],out_l,out_vl);
   LOD_8 #(N>>1, S-1) h(in[N-1:N>>1],out_h,out_vh);
   assign vld = out_vl | out_vh;
   assign out = out_vh ? {1'b0,out_h} : {out_vl,out_l};
   //

endmodule

module LOD_8 (in, out, vld);

parameter N = 32;
parameter S = 5;

   input [N-1:0] in;
   output [S-1:0] out;
   output vld;
   //
   wire [S-2:0] out_l, out_h;
   wire out_vl, out_vh;
   LOD_4 #(N>>1, S-1) l(in[(N>>1)-1:0],out_l,out_vl);
   LOD_4 #(N>>1, S-1) h(in[N-1:N>>1],out_h,out_vh);
   assign vld = out_vl | out_vh;
   assign out = out_vh ? {1'b0,out_h} : {out_vl,out_l};
   //
/*   generate
	begin 
		if (N == 2)
		begin
	  assign vld = |in;
	  assign out = ~in[1] & in[0];
		end
	  else if (N & (N-1))
		//LOD #(1<<S) LOD ({1<<S {1'b0}} | in,out,vld);
		LOD #(1<<S) LOD ({in,{((1<<S) - N) {1'b0}}},out,vld);
	  else
		begin
	  wire [S-2:0] out_l, out_h;
	  wire out_vl, out_vh;
	  LOD #(N>>1) l(in[(N>>1)-1:0],out_l,out_vl);
	  LOD #(N>>1) h(in[N-1:N>>1],out_h,out_vh);
	  assign vld = out_vl | out_vh;
	  assign out = out_vh ? {1'b0,out_h} : {out_vl,out_l};
		end
	end
  endgenerate */
endmodule
module LOD_4 (in, out, vld);
parameter N = 32;
parameter S = 5;

   input [N-1:0] in;
   output [S-1:0] out;
   output vld;
   wire [S-2:0] out_l, out_h;
   wire out_vl, out_vh;
   LOD_2 #(N>>1, S-1) l(in[(N>>1)-1:0],out_l,out_vl);
   LOD_2 #(N>>1, S-1) h(in[N-1:(N>>1)],out_h,out_vh);
   assign vld = out_vl | out_vh;
   assign out = out_vh ? {1'b0,out_h} : {out_vl,out_l};
endmodule
module LOD_2 (in, out, vld);
parameter N = 32;
parameter S = 5;

   input [N-1:0] in;
   output [S-1:0] out;
   output vld;
   //
   assign vld = |in;
   assign out = ~in[1] & in[0];
   //
endmodule

//____________________________________________________________________________________
// =============== IEEE754 module =======================
//____________________________________________________________________________________
module PENC64 (
    Din,
    Dout,
    valid
);

input [63:0]Din;
output[5:0]Dout;
output valid;

wire [4:0]A, B;
wire v0, v1;

PENC32 P0 (.Din(Din[31:0]), .Dout(B), .valid(v0));
PENC32 P1 (.Din(Din[63:32]), .Dout(A), .valid(v1));

assign Valid = v0|v1;
assign Dout[5] = v1;
assign Dout[4:0] = (Dout[5]) ? A : B;

endmodule

module PENC32 (Din, Dout, valid);
    input [31:0] Din;
    output [4:0] Dout;
    output valid;

    wire [2:0] D0, D1, D2, D3;
    wire v0, v1, v2, v3;

    PENC8 P0 (.Din(Din[ 7: 0]), .Dout(D0), .Valid(v0));
    PENC8 P1 (.Din(Din[15: 8]), .Dout(D1), .Valid(v1));
    PENC8 P2 (.Din(Din[23:16]), .Dout(D2), .Valid(v2));
    PENC8 P3 (.Din(Din[31:24]), .Dout(D3), .Valid(v3));

    assign valid = v0|v1|v2|v3;
    assign Dout[4] = v3|v2;
    assign Dout[3] = v3|( !v2 & v1 );
    assign Dout[2:0] = Dout[4]? ((Dout[3])?D3:D2) : ((Dout[3]?D1:D0));
    
endmodule
////////////////////////////////////////////////
module PENC4 (
    Din,
    Dout,
    Valid
);
    
input [3:0]Din;
output[1:0]Dout;
output Valid;

assign Dout[1] = Din[3] | Din[2];
assign Dout[0] = Din[3] | (Din[1] & (!Din[2]));
assign Valid = |Din;

endmodule


////////////////////////////////////////////////
module PENC8 (
    Din,
    Dout,
    Valid
);

input [7:0]Din;
output[2:0]Dout;
output Valid;

wire [1:0]A, B;
wire v0, v1;

PENC4 P0 (.Din(Din[3:0]), .Dout(B), .Valid(v0));
PENC4 P1 (.Din(Din[7:4]), .Dout(A), .Valid(v1));

assign Valid = v0|v1;
assign Dout[2] = v1;
assign Dout[1:0] = (Dout[2]) ? A : B;

endmodule
///////////////////////////////////////////////




//Converter
////////////////////////////////////////////////
module Posit_to_FP (F_P, in, out);

function [31:0] log2;
input reg [31:0] value;
	begin
	value = value-1;
	for (log2=0; value>0; log2=log2+1)
        	value = value>>1;
      	end
endfunction

parameter N = 16;
parameter E = 5;
parameter es = 2;

parameter M = N-E-1;
parameter BIAS = (2**(E-1))-1;
parameter Bs = log2(N); 
parameter EO = E > es+Bs ? E : es+Bs;

input F_P;
input [N-1:0] in;
output [N-1:0] out;

wire s = in[N-1];
wire zero_tmp = |in[N-2:0];
wire inf_in = in[N-1] & (~zero_tmp);
wire zero_in = ~(in[N-1] | zero_tmp);

//Data Extraction
wire rc;
wire [Bs-1:0] rgm, Lshift;
wire [es-1:0] e;
wire [N-es-1:0] mant;
wire [N-1:0] xin = s ? -in : in;
data_extract #(.N(N),.es(es)) uut_de1(.in(xin), .rc(rc), .regime(rgm), .exp(e), .mant(mant), .Lshift(Lshift));

wire [N-1:0] m = {zero_tmp,mant,{es-1{1'b0}}};

//Exponent and Regime Computation
wire [EO+1:0] e_o;
assign e_o = {(rc ? {{EO-es-Bs+1{1'b0}},rgm} : -{{EO-es-Bs+1{1'b0}},rgm}),e} + BIAS;
//Final Output
assign out = F_P ? in : (inf_in|e_o[EO:E]|&e_o[E-1:0] ? {s,{E-1{1'b1}},{M{1'b0}}} : (zero_in|(~m[N-1]) ? {s,{E-1{1'b0}},m[N-2:E]} : { s, e_o[E-1:0], m[N-2:E]} )) ;

endmodule
////////////////////////////////////////////////


module data_extract(in, rc, regime, exp, mant, Lshift);

function [31:0] log2;
input reg [31:0] value;
	begin
	value = value-1;
	for (log2=0; value>0; log2=log2+1)
        	value = value>>1;
      	end
endfunction

parameter N=16;
parameter Bs=log2(N);
parameter es = 2;
input [N-1:0] in;
output rc;
output [Bs-1:0] regime, Lshift;
output [es-1:0] exp;
output [N-es-1:0] mant;

wire [N-1:0] xin = in;
assign rc = xin[N-2];
wire [Bs-1:0] k0, k1;
LOD_N #(.N(N)) xinst_k0(.in({xin[N-2:0],1'b0}), .out(k0));
LZD_N #(.N(N)) xinst_k1(.in({xin[N-3:0],2'b0}), .out(k1));

assign regime = xin[N-2] ? k1 : k0;
assign Lshift = xin[N-2] ? k1+1 : k0;

wire [N-1:0] xin_tmp;
DSR_left_N_S #(.N(N), .S(Bs)) ls (.a({xin[N-3:0],2'b0}),.b(Lshift),.c(xin_tmp));

assign exp= xin_tmp[N-1:N-es];
assign mant= xin_tmp[N-es-1:0];

endmodule

////////////////////////////////////////////////

module LZD_N (in, out);

  function [31:0] log2;
    input reg [31:0] value;
    begin
      value = value-1;
      for (log2=0; value>0; log2=log2+1)
	value = value>>1;
    end
  endfunction

parameter N = 64;
parameter S = log2(N); 
input [N-1:0] in;
output [S-1:0] out;

wire vld;
LZD #(.N(N)) l1 (in, out, vld);
endmodule

module LZD (in, out, vld);

  function [31:0] log2;
    input reg [31:0] value;
    begin
      value = value-1;
      for (log2=0; value>0; log2=log2+1)
	value = value>>1;
    end
  endfunction


parameter N = 64;
parameter S = log2(N);

   input [N-1:0] in;
   output [S-1:0] out;
   output vld;

  generate
    if (N == 2)
      begin
	assign vld = ~&in;
	assign out = in[1] & ~in[0];
      end
    else if (N & (N-1))
      LZD #(1<<S) LZD ({1<<S {1'b0}} | in,out,vld);
    else
      begin
	wire [S-2:0] out_l;
	wire [S-2:0] out_h;
	wire out_vl, out_vh;
	LZD #(N>>1) l(in[(N>>1)-1:0],out_l,out_vl);
	LZD #(N>>1) h(in[N-1:N>>1],out_h,out_vh);
	assign vld = out_vl | out_vh;
	assign out = out_vh ? {1'b0,out_h} : {out_vl,out_l};
      end
  endgenerate
endmodule

////////////////////////////////////////////////

module FP_to_posit(F_P, in, out);

function [31:0] log2;
input reg [31:0] value;
	begin
	value = value-1;
	for (log2=0; value>0; log2=log2+1)
        	value = value>>1;
      	end
endfunction

parameter N = 16;
parameter E = 5;
parameter es = 2;	//ES_max = E-1
parameter M = N-E-1;
parameter BIAS = (2**(E-1))-1;

parameter Bs = log2(N);

input F_P;
input [N-1:0] in;
output [N-1:0] out;

wire s_in = in[N-1];
wire [E-1:0] exp_in = in[N-2:N-1-E];
wire [M-1:0] mant_in = in[M-1:0];
wire zero_in = ~|{exp_in,mant_in};
wire inf_in = &exp_in;

wire [M:0] mant = {|exp_in, mant_in};

wire [N-1:0] LOD_in = {mant,{E{1'b0}}};
wire[Bs-1:0] Lshift;
LOD_N #(.N(N)) uut (.in(LOD_in), .out(Lshift));

wire[N-1:0] mant_tmp;
DSR_left_N_S #(.N(N), .S(Bs)) ls (.a(LOD_in),.b(Lshift),.c(mant_tmp));

wire [E:0] exp = {exp_in[E-1:1], exp_in[0] | (~|exp_in)} - BIAS - Lshift;

//Exponent and Regime Computation
wire [E:0] exp_N = exp[E] ? -exp : exp;
wire [es-1:0] e_o = (exp[E] & |exp_N[es-1:0]) ? exp[es-1:0] : exp_N[es-1:0];
wire [E-es-1:0] r_o = (~exp[E] || (exp[E] & |exp_N[es-1:0])) ? {{Bs{1'b0}},exp_N[E-1:es]} + 1'b1 : {{Bs{1'b0}},exp_N[E-1:es]};

//Exponent and Mantissa Packing
wire [2*N-1:0]tmp_o = { {N{~exp[E]}}, exp[E], e_o, mant_tmp[N-2:es]};

//Including Regime bits in Exponent-Mantissa Packing
wire [2*N-1:0] tmp1_o;
wire [Bs-1:0] diff_b;
generate
	if(E-es > Bs) 	assign diff_b = |r_o[E-es-1:Bs] ? {{(Bs-2){1'b1}},2'b01} : r_o[Bs-1:0];
	else 		assign diff_b = r_o;
endgenerate
DSR_right_N_S #(.N(2*N), .S(Bs)) dsr2 (.a(tmp_o), .b(diff_b), .c(tmp1_o));

//Final Output
wire [N-1:0] tmp1_oN = s_in ? -tmp1_o[N-1:0] : tmp1_o[N-1:0];
assign out = (!F_P)?  in : (inf_in|zero_in|(~mant_tmp[N-1]) ? {inf_in,{N-1{1'b0}}} : {s_in, tmp1_oN[N-1:1]});

endmodule

////////////////////////////////////////////////
